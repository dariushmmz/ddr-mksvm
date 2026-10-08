"""Machine-readable summary of frozen and reserved experiment seed ranges."""

SEED_REGISTRY = {
    "rkhs_gate8": range(7000, 7008),
    "rkhs_confirmation": range(8000, 8096),
    "class_sensitive_gate8": range(9000, 9008),
    "class_sensitive_confirmation": range(10000, 10096),
    "expanded_smoke": range(16000, 16001),
    "expanded_gate8": range(16100, 16108),
    "expanded_gate24": range(16200, 16224),
    "expanded_confirmation": range(16300, 16396),
    "robust_reserved_confirmation": range(14000, 14096),
    "robust_reserved_gate24": range(15400, 15424),
}

RESERVED_RANGES = {"robust_reserved_confirmation", "robust_reserved_gate24"}
