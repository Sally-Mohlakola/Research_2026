
DIAMOND_VARIANTS= {
    "round_diamond_gia": dict(
        girdle_radius=1.0,
        crown_angle_deg=34.5,
        pavilion_angle_deg=40.75,
        table_frac=0.56,
        num_main_facets=8,
        culet_radius=0.02,
        int_ior=2.419,
        ext_ior=1.000277,
    ),
    # Shallow crown / deep pavilion variant -- more "fire", less "scintillation"
    # contrast, included as a second preset to sanity-check the parameter
    # plumbing end-to-end (not a claim about which is more attractive).
    "round_diamond_deep": dict(
        girdle_radius=1.0,
        crown_angle_deg=30.0,
        pavilion_angle_deg=43.5,
        table_frac=0.52,
        num_main_facets=8,
        culet_radius=0.0,  # sharp-point culet
        int_ior=2.419,
        ext_ior=1.000277,
    ),
    # Idealized sharp-culet version of the standard cut, useful for
    # comparing against round_brilliant_ideal to see how much the small
    # flat culet facet patch actually changes the RDM.
    "round_diamond_sharp_culet": dict(
        girdle_radius=1.0,
        crown_angle_deg=34.5,
        pavilion_angle_deg=40.75,
        table_frac=0.56,
        num_main_facets=8,
        culet_radius=0.0,
        int_ior=2.419,
        ext_ior=1.000277,
    ),
}


PEAR_DEFAULTS = {
    "cut": "pear",
    "girdle_radius": 1.0,
    "crown_angle_deg": 34.5,
    "pavilion_angle_deg": 40.75,
    "table_frac": 0.56,
    # 16 girdle points gives 62 triangles, the closest match to the 64 of
    # round_diamond_gia, so a cut ablation changes the outline and not the
    # facet count. Triangles are 4n-2, so no n gives exactly 64.
    "num_girdle_points": 16,
    "taper": 1.0,
    "int_ior": 2.419,
    "ext_ior": 1.000277,
}
DIAMOND_VARIANTS["pear_brilliant"] = PEAR_DEFAULTS


STEP_DEFAULTS = {
    "cut": "step",
    "girdle_radius": 1.0,
    "crown_angle_deg": 34.5,
    "pavilion_angle_deg": 40.75,
    "table_frac": 0.56,
    "length_ratio": 1.35,
    "corner_frac": 0.18,
    "crown_steps": 3,
    "pavilion_steps": 3,
    "keel_ring_frac": 0.30,
    "keel_length_frac": 0.18,
    "step_angle_spread": 0.35,
    "int_ior": 2.419,
    "ext_ior": 1.000277,
}
DIAMOND_VARIANTS["step_cut"] = STEP_DEFAULTS

# Same cut at the one tessellation that lands on exactly 34 vertices and 64
# triangles, matching round_diamond_gia. Face counts are 16*(crown+pavilion)+16,
# so crown_steps+pavilion_steps == 3 is the only way to hit 64. This variant
# exists so a round-trained operator can be re-pointed at step geometry without
# changing the size of its facet categorical: the transfer experiment needs the
# head shapes to agree, and 2+1 keeps a genuinely stepped crown while doing so.
DIAMOND_VARIANTS["step_cut_matched"] = dict(STEP_DEFAULTS, crown_steps=2, pavilion_steps=1)


def get_diamond_parameters(name: str) -> dict:
    if name not in DIAMOND_VARIANTS:
        valid = ", ".join(sorted(DIAMOND_VARIANTS.keys()))
        raise ValueError(f"Unknown diamond variant '{name}'. Valid options: {valid}")
    return dict(DIAMOND_VARIANTS[name]) 