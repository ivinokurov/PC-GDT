"""PC-GDT: physics-conditioned generative digital twin (runnable reference pipeline)."""
__version__ = "1.0.0"
from .geometry import (Scene, RoofFace, Canopy, sun_direction, sun_embedding,
                       antisolar_azimuth, angular_difference_deg,
                       illuminated_faces, random_scenes, make_gabled_house,
                       make_flat_house, project_point_on_face,
                       shadow_points_on_face, shadow_of_canopy_on_face,
                       ridge_break_of_shadow)
from .conditions import ConditionMaps, rasterise_scene, CLASS_NAMES, N_CLASSES
from .model import PCControlNet, FiLMSunModulator, pc_gdt_loss, sample, cosine_lr_with_warmup
from .violations import ViolationMeasures, compute_violations
from .pgpf import (PGPFThresholds, SampleMeasurements, run_pgpf,
                   posterior_defect_rate, expected_acceptance,
                   solve_total_false_rejection, total_variation_bound)