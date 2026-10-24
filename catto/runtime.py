"""Window branching and selection around SeeSR's unchanged noise prediction."""
from dataclasses import asdict
import hashlib
from pathlib import Path

import torch
from PIL import Image

from .condition import DCTParameterReuseState
from .scoring import output_aligned_images
from .selection import select


def repeat_embeddings(value, count, cfg):
    if value is None:
        return None
    if cfg:
        return torch.cat([value[:1].expand(count, *value.shape[1:]),
                          value[1:2].expand(count, *value.shape[1:])], dim=0)
    return value[:1].expand(count, *value.shape[1:])


def tensor_digest(value):
    return hashlib.sha256(value.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()


class WindowState:
    """One state per pipeline call; never reused between input images."""
    def __init__(self, pipe, config, image, latents, cfg, guess_mode,
                 num_steps, tile_size, start_steps, seed, align_method):
        config.validate_schedule(num_steps)
        if image.shape[0] != (2 if cfg else 1) or latents.shape[0] != 1:
            raise ValueError("CATTO currently supports one input image per pipeline call")
        if guess_mode:
            raise ValueError("CATTO currently requires guess_mode=False")
        if latents.shape[-2] * latents.shape[-1] > tile_size * tile_size:
            raise ValueError("CATTO windows require an untiled latent; use a crop or a larger latent_tiled_size")
        if type(pipe.scheduler).__name__ != "DDPMScheduler":
            raise ValueError("CATTO uses SeeSR's original DDPMScheduler")
        if any(int(t) > start_steps for t in pipe.scheduler.timesteps):
            raise ValueError("CATTO cannot use a schedule with skipped denoising steps")
        self.pipe, self.config, self.cfg = pipe, config, cfg
        self.seed, self.align_method = seed or 0, align_method
        self.fixed = image[:1].detach().clone()
        self.reuse = DCTParameterReuseState(self.fixed, config.condition_reuse, config.reuse_decay)
        self.by_start = {w.start: w for w in config.windows}
        self.event = self.reference = self.reference_step = None
        self.active_reference = self.event_conditions = None
        self.records, self.condition_records = [], []
        self.nfe, self.num_steps, self.num_calls = 0, num_steps, 0

    def model_image(self, condition):
        return torch.cat([condition] * 2) if self.cfg else condition

    def prepare(self, step, latents, image, prompt, ram):
        event = self.by_start.get(step)
        if event is None and self.event is None:
            working = self.reuse.advance()
            if working is not None:
                image = self.model_image(working)
        if event is not None:
            if self.event is not None or latents.shape[0] != 1:
                raise RuntimeError("overlapping CATTO window or uncollapsed candidate batch")
            self.event = event
            self.active_reference = (self.reference if self.reference_step == step - 1 else self.fixed)
            self.active_reference_step = self.reference_step if self.reference_step == step - 1 else None
            event_rng = torch.Generator(device=latents.device).manual_seed(self.seed + 1009 * step)
            self.event_conditions = self.reuse.begin_event(event.candidates, event.scale, event_rng)
            self.base_prompt, self.base_ram = prompt, ram
            latents = latents[:1].expand(event.candidates, -1, -1, -1).clone()
            image = self.model_image(self.event_conditions)
            prompt = repeat_embeddings(prompt, event.candidates, self.cfg)
            ram = repeat_embeddings(ram, event.candidates, self.cfg)
            self.trajectory = torch.zeros(event.candidates, device=latents.device, dtype=torch.float32)
            self.trajectory_steps = 0
        return latents, image, prompt, ram

    @torch.no_grad()
    def decode(self, clean_latents):
        images = []
        for chunk in clean_latents.split(self.config.decode_chunk_size):
            decoded = self.pipe.vae.decode(chunk / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
            images.append((decoded / 2 + 0.5).clamp(0, 1).float())
        return torch.cat(images, dim=0)

    def step(self, step, timestep, noise, latents, image, prompt, ram, extra_step_kwargs):
        self.nfe += latents.shape[0]
        self.num_calls += 1
        event = self.event
        ending = event is not None and step == event.end
        before_window = event is None and step + 1 in self.by_start
        if event is not None:
            self.trajectory += (noise.float() - noise[:1].float()).square().flatten(1).mean(1)
            self.trajectory_steps += 1
        if ending or before_window:
            output = self.pipe.scheduler.step(noise, timestep, latents, **extra_step_kwargs, return_dict=True)
            if getattr(output, "pred_original_sample", None) is None:
                raise RuntimeError("DDPM must provide pred_original_sample for x0 scoring")
            latents = output.prev_sample
            clean_images = self.decode(output.pred_original_sample)
        else:
            latents = self.pipe.scheduler.step(noise, timestep, latents, **extra_step_kwargs, return_dict=False)[0]
        if ending:
            reward_images = output_aligned_images(clean_images, self.fixed, self.align_method)
            rewards = self.pipe.catto_reward_model(reward_images.float()).reshape(-1).float()
            selected, record = select(clean_images, rewards, self.event_conditions,
                                      self.trajectory / self.trajectory_steps, self.active_reference,
                                      event.mode, self.config.consistency_weights,
                                      self.config.topk, self.config.mse_floor)
            # Keep the selected latent trajectory. Reuse carries selected DCT
            # parameters, not decoded images, into later window proposals.
            latents = latents[selected:selected + 1]
            working = self.reuse.commit_selection(selected)
            image = self.model_image(working)
            prompt, ram = self.base_prompt, self.base_ram
            record.update(window=asdict(event), step=step, timestep=int(timestep),
                          selection_policy=event.policy, reference_source="pre_event_x0",
                          reference_source_step=self.active_reference_step,
                          actual_reference_sha256=tensor_digest(self.active_reference),
                          selected_raw_x0_sha256=tensor_digest(clean_images[selected:selected + 1]),
                          condition_reuse=self.config.condition_reuse,
                          reuse_decay=self.config.reuse_decay, **self.reuse.diagnostics())
            self.records.append(record)
            self.reference = clean_images[selected:selected + 1].detach().clone()
            self.reference_step = step
            self.event = self.event_conditions = self.active_reference = None
            print(f"[CATTO] window={event.start}:{event.length} mode={event.mode} selected={selected}", flush=True)
        elif before_window:
            self.reference, self.reference_step = clean_images[:1].detach().clone(), step
        if self.config.save_conditions:
            self.save_condition(step, timestep, ending)
        return latents, image, prompt, ram

    def save_condition(self, step, timestep, ending):
        # Exactly one committed condition per step, even inside a search window.
        # Temporary candidate conditions are intentionally not exported.
        condition = self.reuse.condition.detach().float().clamp(0, 1).cpu()
        folder = Path(self.pipe._catto_sample_dir) / "optimized_conditions"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"step_{step:02d}.png"
        rgb = (condition[0].permute(1, 2, 0).numpy() * 255 + 0.5).astype("uint8")
        Image.fromarray(rgb, mode="RGB").save(target)
        self.condition_records.append(dict(step=step, timestep=int(timestep),
                                           window_end=ending, png=str(target.name),
                                           sha256=tensor_digest(condition)))

    def finish(self):
        expected = self.config.nfe(self.num_steps)
        if self.event is not None or self.nfe != expected or self.num_calls != self.num_steps:
            raise RuntimeError(f"CATTO schedule audit failed: {self.nfe} NFE, expected {expected}")
        self.pipe.catto_diagnostics = dict(
            candidate_nfe=self.nfe, unet_forward_steps=self.num_calls,
            config=self.config.to_dict(), events=self.records,
            conditions=self.condition_records,
        )
