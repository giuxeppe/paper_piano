import numpy as np
import sounddevice as sd
import threading
import time

NOTE_TO_SEMITONE = {
    "C": 0,  "C#": 1,
    "D": 2,  "D#": 3,
    "E": 4,
    "F": 5,  "F#": 6,
    "G": 7,  "G#": 8,
    "A": 9,  "A#": 10,
    "B": 11
}

def note_name_to_midi(name: str) -> int:
    name = name.strip()
    if len(name) >= 2 and name[1] == "#":
        pitch = name[:2]
        octave = int(name[2:])
    else:
        pitch = name[:1]
        octave = int(name[1:])
    return (octave + 1) * 12 + NOTE_TO_SEMITONE[pitch]  

def midi_to_freq(m: int) -> float:
    return 440.0 * (2.0 ** ((m - 69) / 12.0))

def _softclip(x):
    # gentle saturation to avoid harsh clipping
    return np.tanh(x)

class PianoLikeSynth:
    """
    Piano-ish additive synth:
      - Multiple harmonics (partials) with individual decay rates
      - Bright attack (harmonics decay faster than fundamental)
      - Optional hammer noise burst at note-on
      - Note off shortens tail (damping), but not instant silence

    API:
      synth.note_on("C4", velocity=1.0)
      synth.note_off("C4")
    """
    def __init__(
        self,
        sample_rate=48000,
        blocksize=256,
        max_poly=16,
        volume=0.25,
        attack=0.002,          
        release=0.20,          
        sustain_decay=1.8,     
        brightness=0.9,        
        hammer_amount=0.25,    
        hammer_ms=12.0         
    ):
        self.sr = int(sample_rate)
        self.block = int(blocksize)
        self.max_poly = int(max_poly)
        self.volume = float(volume)

        self.attack = float(attack)
        self.release = float(release)
        self.sustain_decay = float(sustain_decay)
        self.brightness = float(brightness)
        self.hammer_amount = float(hammer_amount)
        self.hammer_ms = float(hammer_ms)

        self.notes = {}  
        self.lock = threading.Lock()

        self.stream = sd.OutputStream(
            samplerate=self.sr,
            blocksize=self.block,
            channels=1,
            dtype="float32",
            callback=self._callback
        )
        self.stream.start()

    def close(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass

    def note_on(self, name: str, velocity: float = 1.0):
        freq = midi_to_freq(note_name_to_midi(name))
        velocity = float(np.clip(velocity, 0.0, 1.0))

        with self.lock:
            if len(self.notes) >= self.max_poly and name not in self.notes:
                self.notes.pop(next(iter(self.notes)))

            base_partials = 8
            extra = int(8 * np.clip(self.brightness * (0.6 + 0.4 * velocity), 0.0, 1.0))
            n_partials = int(np.clip(base_partials + extra, 6, 16))

            # harmonic amplitudes (rough piano-ish): 1/k with some shaping
            ks = np.arange(1, n_partials + 1, dtype=np.float32)
            amps = (1.0 / ks)
            # brighten: emphasize upper harmonics a bit
            amps *= (1.0 + 0.6 * self.brightness * (ks / n_partials))
            # normalize partial mix
            amps = amps / (np.sum(amps) + 1e-9)

            # per-partial decay: higher harmonics decay faster
            # decay_rate per sample: exp(-t/tau)
            # tau seconds: fundamental longer, highs shorter
            tau_fund = 1.4 * self.sustain_decay  # seconds-ish
            tau_high = 0.25 * self.sustain_decay
            taus = tau_fund - (tau_fund - tau_high) * ((ks - 1) / max(1, n_partials - 1))
            taus = np.clip(taus, 0.05, 10.0)
            decay_per_sample = np.exp(-1.0 / (taus * self.sr)).astype(np.float32)

            st = self.notes.get(name)
            if st is None:
                st = {
                    "freq": freq,
                    "phases": np.zeros(n_partials, dtype=np.float32),
                    "amps": amps.astype(np.float32),
                    "decay": decay_per_sample,
                    "env": 0.0,          
                    "held": True,
                    "vel": velocity,
                    "age": 0,            
                    "hammer_left": int((self.hammer_ms / 1000.0) * self.sr),
                }
                self.notes[name] = st
            else:
                # re-trigger: reset age/hammer, keep new partials
                st["freq"] = freq
                st["phases"] = np.zeros(n_partials, dtype=np.float32)
                st["amps"] = amps.astype(np.float32)
                st["decay"] = decay_per_sample
                st["env"] = 0.0
                st["held"] = True
                st["vel"] = velocity
                st["age"] = 0
                st["hammer_left"] = int((self.hammer_ms / 1000.0) * self.sr)

    def note_off(self, name: str):
        with self.lock:
            st = self.notes.get(name)
            if st is not None:
                st["held"] = False

    def _callback(self, outdata, frames, time_info, status):
        t = np.arange(frames, dtype=np.float32)
        mix = np.zeros(frames, dtype=np.float32)

        with self.lock:
            items = list(self.notes.items())

        if not items:
            outdata[:, 0] = 0.0
            return

        # attack step
        a_step = 1.0 / max(1.0, self.attack * self.sr)
        # release/damping step for env when released
        r_step = 1.0 / max(1.0, self.release * self.sr)

        to_delete = []

        with self.lock:
            for name, st in items:
                f0 = st["freq"]
                phases = st["phases"]
                amps = st["amps"]
                decay = st["decay"]
                vel = st["vel"]

                n = len(phases)
                ks = np.arange(1, n + 1, dtype=np.float32)

                # phase increment per partial
                dphi = (2.0 * np.pi * f0 * ks / self.sr).astype(np.float32)

                # build oscillators (vectorized over partials, then sum)
                # phases shape (n,), we need (n,frames): phases[:,None] + dphi[:,None]*t[None,:]
                ph = phases[:, None] + dphi[:, None] * t[None, :]
                sig = np.sin(ph, dtype=np.float32)

                # per-partial exponential amplitude decay over time (relative to note age)
                # Instead of tracking separate per-sample decay, update partial amps by decay^frames each block.
                # For within-block shaping, apply a small ramp approximation:
                # amp_block[k] * decay[k]^(sample_index)
                sample_idx = np.arange(frames, dtype=np.float32)[None, :]
                dec_curve = np.power(decay[:, None], sample_idx, dtype=np.float32)
                partial_env = amps[:, None] * dec_curve

                # attack envelope (shared)
                env = st["env"]
                if st["held"]:
                    ramp = env + a_step * np.arange(frames, dtype=np.float32)
                    env_curve = np.minimum(ramp, 1.0)
                    env_end = float(env_curve[-1])
                else:
                    ramp = env - r_step * np.arange(frames, dtype=np.float32)
                    env_curve = np.maximum(ramp, 0.0)
                    env_end = float(env_curve[-1])

                # hammer noise burst (short)
                hammer = 0.0
                hl = int(st.get("hammer_left", 0))
                if hl > 0 and self.hammer_amount > 0:
                    n_h = min(frames, hl)
                    noise = (np.random.randn(n_h).astype(np.float32))
                    # quick high-pass-ish by differencing (brighter "click")
                    noise[1:] = noise[1:] - 0.8 * noise[:-1]
                    # fade out burst
                    fade = np.linspace(1.0, 0.0, n_h, dtype=np.float32)
                    hammer_buf = np.zeros(frames, dtype=np.float32)
                    hammer_buf[:n_h] = noise * fade
                    hammer = self.hammer_amount * vel * hammer_buf
                    st["hammer_left"] = hl - n_h

                # sum partials
                block = np.sum(sig * partial_env, axis=0)

                # apply shared env and velocity
                block = block * env_curve * vel

                # add hammer
                block = block + hammer

                # accumulate
                mix += block

                # update partial amps for next block: amps *= decay^frames
                decay_block = np.power(decay, frames, dtype=np.float32)
                st["amps"] = (amps * decay_block).astype(np.float32)

                # update phases at end of block
                phases_end = (phases + dphi * frames) % (2.0 * np.pi)
                st["phases"] = phases_end.astype(np.float32)

                # update env
                st["env"] = env_end

                # age
                st["age"] += frames

                # remove if silent enough
                if (not st["held"]) and env_end <= 1e-4:
                    # also if partial energy is tiny
                    if float(np.sum(st["amps"])) < 1e-4:
                        to_delete.append(name)

            for name in to_delete:
                self.notes.pop(name, None)

        # soft normalization by polyphony
        denom = max(1.0, np.sqrt(len(items)))
        mix = (self.volume / denom) * mix

        # soft clip
        mix = _softclip(mix)

        outdata[:, 0] = mix.astype(np.float32)

if __name__ == "__main__":
    synth = PianoLikeSynth(volume=0.30, brightness=0.9, hammer_amount=0.35)
    try:
        synth.note_on("C4", 1.0); time.sleep(0.25)
        synth.note_on("E4", 1.0); time.sleep(0.25)
        synth.note_on("G4", 1.0); time.sleep(0.25)
        synth.note_on("B4", 1.0); time.sleep(0.60)
        synth.note_off("C4"); synth.note_off("E4"); synth.note_off("G4"); synth.note_off("E4")
        time.sleep(0.8)
    finally:
        synth.close()