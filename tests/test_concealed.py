from floorfathom.concealed import concealed_flags
from floorfathom.schema import DamageRegion, Measurement, RoomPlan, SurfaceRef, Wall


def _m(v, unit="m"):
    return Measurement(value=v, lo=v * 0.9, hi=v * 1.1, unit=unit, method="test")


def _room(rid="room_0", square=(0.0, 0.0, 4.0, 3.0)):
    x0, y0, x1, y1 = square
    poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    walls = [
        Wall(id=f"{rid}_w{i}", start=a, end=b, length=_m(1.0), evidence="wall_points")
        for i, (a, b) in enumerate(zip(poly, poly[1:] + poly[:1]))
    ]
    return RoomPlan(id=rid, polygon=poly, walls=walls, ceiling_height=_m(2.6), floor_area=_m(12.0, "m2"), openings=[])


def _dmg(did, cls, kind, sid, centre, length=0.2):
    return DamageRegion(
        id=did, surface=SurfaceRef(kind=kind, id=sid), damage_class=cls, class_confidence=0.8,
        polygon=[centre] * 3, centre=centre, width=_m(length), height=_m(length), length=_m(length), area=_m(length * length, "m2"), evidence="test",
    )


def _rules(flags):
    return sorted(f.rule for f in flags)


def test_each_rule_fires_on_its_own_case_and_names_itself():
    room = _room()
    cases = [
        (_dmg("d", "water_stain", "ceiling", "room_0_ceiling", (1.0, 1.0)), "ceiling_stain_source_above"),
        (_dmg("d", "mould", "wall", "room_0_w0", (1.0, 1.5)), "mould_moisture_behind_surface"),
        (_dmg("d", "efflorescence", "wall", "room_0_w0", (1.0, 1.5)), "moisture_migrating_through_coating"),
        (_dmg("d", "structural_crack", "wall", "room_0_w0", (1.0, 1.5), length=0.6), "long_crack_continues_behind_finish"),
        (_dmg("d", "sagging_or_bulging", "ceiling", "room_0_ceiling", (2.0, 1.5)), "ceiling_sag_water_or_load"),
        (_dmg("d", "soot_or_fire", "wall", "room_0_w0", (1.0, 1.5)), "soot_char_behind_surface"),
    ]
    for d, rule in cases:
        got = concealed_flags(room, [d], [])
        assert rule in _rules(got), (rule, _rules(got))
        assert all(f.damage_ids == ["d"] and f.reason for f in got)


def test_short_crack_and_high_dry_stain_flag_nothing():
    room = _room()
    assert concealed_flags(room, [_dmg("a", "structural_crack", "wall", "room_0_w0", (1.0, 1.5), length=0.2)], []) == []
    assert concealed_flags(room, [_dmg("b", "hole_or_impact", "wall", "room_0_w0", (1.0, 1.5))], []) == []
    assert concealed_flags(room, [], []) == []


def test_stain_low_on_a_wall_and_stain_under_a_ceiling_stain():
    room = _room()
    low = _dmg("low", "water_stain", "wall", "room_0_w0", (1.0, 0.3))
    assert "moisture_low_on_wall" in _rules(concealed_flags(room, [low], []))
    ceil = _dmg("ceil", "water_stain", "ceiling", "room_0_ceiling", (1.0, 0.1))  # over wall 0 at x=1
    wall = _dmg("wall", "water_stain", "wall", "room_0_w0", (1.0, 1.8))
    got = concealed_flags(room, [ceil, wall], [])
    under = [f for f in got if f.rule == "wall_stain_under_ceiling_stain"]
    assert under and set(under[0].damage_ids) == {"wall", "ceil"}
    far = _dmg("far", "water_stain", "ceiling", "room_0_ceiling", (3.5, 2.5))
    assert "wall_stain_under_ceiling_stain" not in _rules(concealed_flags(room, [far, wall], []))


def test_stain_on_a_wall_shared_with_another_room():
    room, other = _room(), _room("room_1", (4.2, 0.0, 8.0, 3.0))  # 20 cm of wall between the two
    on_shared = _dmg("s", "water_stain", "wall", "room_0_w1", (1.5, 1.5))  # wall 1 runs x=4 from y 0 to 3
    assert "stain_on_shared_wall" in _rules(concealed_flags(room, [on_shared], [other]))
    away = _dmg("a", "water_stain", "wall", "room_0_w3", (1.5, 1.5))
    assert "stain_on_shared_wall" not in _rules(concealed_flags(room, [away], [other]))


def test_flags_have_unique_ids_and_no_duplicates():
    room = _room()
    got = concealed_flags(room, [_dmg("d", "mould", "wall", "room_0_w0", (1.0, 0.3))], [])
    assert len({f.id for f in got}) == len(got) == 2  # mould, and moisture low on the wall
