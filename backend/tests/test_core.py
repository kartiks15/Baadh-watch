"""Unit tests for the grounding guarantee (no invented numbers) and the cut-off logic."""
import numpy as np
from shapely.geometry import LineString, Point, box

from pipeline import network, sitrep
from pipeline.config import Grid, Thresholds
from pipeline.impact import Affected


def _facts():
    return {"settlements_cut_off": {"value": 3, "desc": "cut off"},
            "people_cut_off": {"value": 1234, "desc": "people"},
            "s1_name": {"value": "Timure", "desc": "name"}, "s1_name_ne": {"value": "टिमुरे", "desc": "name ne"},
            "s1_status": {"value": "cut_off", "desc": "status"}, "_n_priority": 1}


def test_validator_rejects_raw_digits():
    f = _facts()
    assert sitrep.validate("{{settlements_cut_off}} villages are cut off.", f) == []
    assert sitrep.validate("3 villages are cut off.", f)            # ASCII digit
    assert sitrep.validate("३ बस्ती सम्पर्कविहीन", f)               # Devanagari digit
    assert sitrep.validate("{{made_up_key}} people", f)            # unknown key


def test_nepali_quality_floor():
    f = _facts()
    assert sitrep.validate("{{settlements_cut_off}} बस्ती सम्पर्कविहीन", f, "ne") == []
    assert sitrep.validate("अस्पताल पहुँच गँvai लост", f, "ne")      # Cyrillic + mixed-script word
    assert sitrep.validate("अवरुद्ध सडक लENGTH", f, "ne")              # Latin glued to Devanagari
    assert sitrep.validate("OSM डेटा {{s1_name}}", f, "ne") == []     # separate Latin words are fine


def test_render_nepali_digits_and_names():
    f = _facts()
    out = sitrep.render("{{s1_name}}: {{s1_status}}, {{people_cut_off}}", f, "ne")
    assert out == "टिमुरे: सम्पर्कविहीन, १,२३४"
    assert sitrep.render("{{people_cut_off}}", f, "en") == "1,234"


def test_nepali_digits_only_for_quantities_and_dates():
    f = {"model_name": {"value": "U-Net/resnet34 trained on Sen1Floods11", "desc": "model"},
         "event_date": {"value": "2026-08-26", "desc": "date"}, "bridges_damaged": {"value": 59, "desc": "n"},
         "_n_priority": 0}
    out = sitrep.render("{{model_name}} {{event_date}} {{bridges_damaged}}", f, "ne")
    assert out == "U-Net/resnet34 trained on Sen1Floods11 २०२६-०८-२६ ५९"


def test_templates_only_use_known_keys():
    run = {"detect": {"affected_ha": 10.0, "water_ha": 4.0, "debris_ha": 6.0, "high_conf_ha": 5.0,
                      "valley_observed_pct": 80.0},
           "buildings": {"in_water": 1, "in_debris": 2, "at_risk": 3, "ok": 4, "total": 10},
           "bridges": {"total": 2, "likely_damaged": 1}, "roads": {"km_by_class": {"primary": 1.5}},
           "network": {"counts": {"cut_off": 1, "detour": 0, "connected": 1, "no_road_pre": 0, "people_cut_off": 44,
                                  "hospital_access_lost": 1},
                       "settlements": [{"name": "A", "status": "cut_off", "est_people": 44, "pre_km": 3.0,
                                        "post_km": None, "nearest": "Town", "cause": {"km_along_route": 1.0,
                                                                                      "bridge": True}}]},
           "bbox": [85.0, 28.0, 85.1, 28.1], "event_date": "2026-08-26", "pairs": [], "osm_snapshot": "2026-08-01",
           "model": "test", "thresholds": {"building_risk_buffer_m": 30.0, "people_per_building": 4.4}}
    facts = sitrep.build_facts(run)
    for lang in ("en", "ne"):
        tpl = sitrep.template(facts, lang)
        assert sitrep.validate(tpl, facts) == []  # templates obey the same rule as the language model
        assert "{{" not in sitrep.render(tpl, facts, lang)


def test_cut_off_when_only_road_is_flooded():
    """Village V — road — flooded segment — road — Town T.  V must become cut off, W (same side as T) not."""
    th = Thresholds()
    road = LineString([(0, 0), (1000, 0), (2000, 0), (3000, 0)])
    spur = LineString([(3000, 0), (3000, 500)])
    flood = [(box(1400, -50, 1600, 50), {"class": "debris", "confidence": "high"})]
    aff = Affected(flood)
    G, _ = network.build_graph([road, spur], [{"highway": "primary"}, {"highway": "track"}], aff, set(), th)
    places = [Point(0, 10), Point(3000, 510), Point(3000, -5)]
    props = [{"place": "village", "name": "V"}, {"place": "village", "name": "W"}, {"place": "town", "name": "T"}]
    grid = Grid(epsg=32645, res=10, x0=-500, y1=1000, width=400, height=200)
    res = network.analyse(G, places, props, [], [], np.zeros((0, 2)), grid, th)
    status = {r["name"]: r["status"] for r in res["settlements"]}
    assert status["V"] == "cut_off"
    assert status["W"] == "connected"
    v = next(r for r in res["settlements"] if r["name"] == "V")
    assert v["cause"]["km_along_route"] == 1.0  # first blocked segment starts 1 km from V


def test_ohsome_parquet_parsing():
    """Parse a table shaped like ohsome v2 features.parquet (osm_type, osm_id, tags MAP, geom WKB)."""
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq
    import shapely
    from shapely.geometry import Polygon

    from pipeline import ohsome

    poly = Polygon([(79.6, 30.5), (79.61, 30.5), (79.61, 30.51), (79.6, 30.5)])
    table = pa.table({
        "osm_type": ["way", "node"],
        "osm_id": pa.array([11, 22], pa.int64()),
        "tags": pa.array([[("amenity", "hospital"), ("name", "Joshimath CHC")], [("place", "village")]],
                         pa.map_(pa.string(), pa.string())),
        "geom": [shapely.to_wkb(poly), shapely.to_wkb(Point(79.7, 30.55))],
    })
    buf = io.BytesIO()
    pq.write_table(table, buf)
    feats = ohsome._features(pq.read_table(io.BytesIO(buf.getvalue())), "health")
    assert feats[0]["properties"] == {"amenity": "hospital", "name": "Joshimath CHC", "@osmId": "way/11"}
    assert feats[0]["geometry"]["type"] == "Point"  # facility outline reduced to a point
    assert feats[1]["properties"]["@osmId"] == "node/22"
