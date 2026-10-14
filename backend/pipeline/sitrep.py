"""Situation reports and Q&A in English and Nepali, grounded in the pipeline's own numbers.

Every number shown to a rescuer comes from `facts.json`. The language model never writes a number:
it may only reference facts as `{{key}}` placeholders, which this module substitutes. Any output that
contains a raw digit (ASCII or Devanagari) or an unknown key is rejected and retried, then falls back to a
deterministic template. Without an API key the template is used directly.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import requests

from .config import (ANTHROPIC_API_KEY, LLM_MODEL, LLM_PROVIDER, OPENROUTER_API_KEY, OPENROUTER_MAX_TOKENS,
                     OPENROUTER_MODEL)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")
# Greek, Cyrillic, Arabic, Thai, Japanese kana, CJK, Hangul: never valid in a Nepali report
FOREIGN_SCRIPT = re.compile("[Ͱ-ϿЀ-ӿ؀-ۿ฀-๿぀-ヿ一-鿿가-힯]")
# a single word glued from Latin letters and Devanagari (e.g. "लENGTH"), a sign of garbled generation
MIXED_WORD = re.compile("(?:[ऀ-ॿ]+[A-Za-z]+|[A-Za-z]+[ऀ-ॿ]+)[\\wऀ-ॿ]*")
DIGIT = re.compile(r"[0-9०-९]")
NE_DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")
TOP_N = 12
LOCALISED_DIGIT_KEYS = {"event_date", "generated_utc", "osm_snapshot", "aoi"}  # strings shown in Nepali digits

STATUS = {
    "en": {"cut_off": "CUT OFF", "detour": "long detour", "connected": "connected", "no_road_pre": "no road before event"},
    "ne": {"cut_off": "सम्पर्कविहीन", "detour": "लामो घुमाउरो बाटो", "connected": "जोडिएको",
           "no_road_pre": "घटनाअघि पनि सडक नभएको"},
}
FOOTER = {  # appended by code: attributions contain digits the language model is not allowed to write
    "en": "\n\n---\n*Contains modified Copernicus Sentinel data 2026 · Produced using Copernicus WorldDEM-30 © DLR e.V. "
          "2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and "
          "ESA · © OpenStreetMap contributors. Educational prototype, not an operational tool.*",
    "ne": "\n\n---\n*Contains modified Copernicus Sentinel data 2026 · Copernicus WorldDEM-30 © DLR e.V. 2010–2014, "
          "© Airbus Defence and Space GmbH 2014–2018 · © OpenStreetMap contributors. शैक्षिक प्रोटोटाइप, "
          "सञ्चालनका लागि होइन।*",
}
YESNO = {"en": {True: "yes", False: "no"}, "ne": {True: "हो", False: "होइन"}}


def _fmt(v, nd=1):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    return str(v)


def build_facts(run: dict) -> dict:
    """Flat key → value catalogue (+ descriptions) from a finished run's summary."""
    d, b, br, net, roads = run["detect"], run["buildings"], run["bridges"], run["network"], run["roads"]
    c = net["counts"]
    rows = net["settlements"]
    hit = b["in_water"] + b["in_debris"]
    w, s, e, n = run["bbox"]
    f = {}

    def put(key, value, desc):
        f[key] = {"value": value, "desc": desc}

    put("aoi", f"{w:.3f}–{e:.3f}°E, {s:.3f}–{n:.3f}°N", "analysed area (bounding box)")
    put("event_date", run["event_date"], "flood date given by the user")
    put("generated_utc", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "report time")
    put("post_image", "; ".join(p["post"]["label"] for p in run["pairs"]) or "none", "post-event radar image(s)")
    put("pre_images", "; ".join(x["label"] for p in run["pairs"] for x in p["pres"]) or "none",
        "pre-event radar image(s), same orbit track")
    put("optical", run.get("s2_note", "not used"), "Sentinel-2 optical use")
    put("osm_snapshot", run["osm_snapshot"], "OpenStreetMap snapshot date (pre-event)")
    put("model_name", run["model"], "water segmentation model")
    put("affected_ha", round(d["affected_ha"], 1), "flooded + debris-covered area detected (hectares)")
    put("water_ha", round(d["water_ha"], 1), "area classed as water (ha)")
    put("debris_ha", round(d["debris_ha"], 1), "area classed as debris / changed ground (ha)")
    put("high_conf_ha", round(d["high_conf_ha"], 1), "affected area confirmed by ≥2 independent sources (ha)")
    put("valley_observed_pct", round(d["valley_observed_pct"], 1), "% of valley floor observable by satellites")
    put("buildings_total", b["total"], "pre-event OSM buildings in the area")
    put("buildings_hit", hit, "buildings inside detected water or debris")
    put("buildings_in_water", b["in_water"], "buildings inside detected water")
    put("buildings_in_debris", b["in_debris"], "buildings inside detected debris")
    put("buildings_at_risk", b["at_risk"], "further buildings within the risk buffer")
    put("risk_buffer_m", int(run["thresholds"]["building_risk_buffer_m"]), "risk buffer around affected areas (m)")
    put("people_in_hit_buildings", int(round(hit * run["thresholds"]["people_per_building"])),
        "estimated people living in hit buildings (proxy)")
    put("road_km_blocked", round(sum(roads["km_by_class"].values()), 1), "road length likely blocked (km)")
    put("bridges_total", br["total"], "road bridges in the area (OSM)")
    put("bridges_damaged", br["likely_damaged"], "bridges likely damaged / washed out")
    put("settlements_total", len(rows), "named settlements in the area")
    put("settlements_cut_off", c["cut_off"], "settlements that lost road connection to nearest town/hospital")
    put("settlements_detour", c["detour"], "settlements still connected but with a much longer route")
    put("settlements_connected", c["connected"], "settlements still connected normally")
    put("settlements_no_road", c["no_road_pre"], "settlements with no road connection even before the event")
    put("people_cut_off", c["people_cut_off"], "estimated people in cut-off settlements (proxy)")
    put("hospital_access_lost", c["hospital_access_lost"], "settlements that lost road access to every hospital")
    put("people_per_building", run["thresholds"]["people_per_building"], "assumed people per building")

    pri = [r for r in rows if r["status"] in ("cut_off", "detour")][:TOP_N]
    for i, r in enumerate(pri, 1):
        cause = r.get("cause") or {}
        put(f"s{i}_name", r.get("name_en") or r["name"], f"priority settlement {i}: name")
        put(f"s{i}_name_ne", r.get("name_ne") or r["name"], f"priority settlement {i}: Nepali name")
        put(f"s{i}_status", r["status"], f"priority settlement {i}: status")
        put(f"s{i}_people", r["est_people"], f"priority settlement {i}: estimated people")
        put(f"s{i}_pre_km", r["pre_km"], f"priority settlement {i}: road km to {r['nearest']} before")
        put(f"s{i}_post_km", r["post_km"], f"priority settlement {i}: road km after (n/a = no route)")
        put(f"s{i}_nearest", r["nearest"], f"priority settlement {i}: nearest town/hospital before the event")
        put(f"s{i}_break_km", cause.get("km_along_route"), f"priority settlement {i}: km to first blocked point")
        put(f"s{i}_break_bridge", bool(cause.get("bridge")), f"priority settlement {i}: first break is a bridge")
    f["_n_priority"] = len(pri)
    return f


def render(text: str, facts: dict, lang: str) -> str:
    def sub(m):
        k = m.group(1)
        v = facts[k]["value"]
        if k.endswith("_status"):
            return STATUS[lang].get(v, v)
        if isinstance(v, bool):
            return YESNO[lang][v]
        if k.endswith("_name") and lang == "ne" and f"{k}_ne" in facts:
            return str(facts[f"{k}_ne"]["value"])
        out = _fmt(v)
        # quantities, dates and coordinates in Devanagari digits; identifiers (model, image labels) unchanged
        numeric = isinstance(v, (int, float)) or k in LOCALISED_DIGIT_KEYS
        return out.translate(NE_DIGITS) if lang == "ne" and numeric else out

    return PLACEHOLDER.sub(sub, text)


def validate(text: str, facts: dict, lang: str = "en") -> list[str]:
    problems = []
    if lang == "ne":  # quality floor for Nepali: no foreign scripts, no words mixing Latin and Devanagari
        bad = FOREIGN_SCRIPT.findall(text) + MIXED_WORD.findall(PLACEHOLDER.sub(" ", text))
        if bad:
            problems.append(f"Nepali text contains foreign-script or mixed-script words: {bad[:5]}")
    unknown = sorted({k for k in PLACEHOLDER.findall(text) if k not in facts})
    if unknown:
        problems.append(f"unknown placeholder keys: {', '.join(unknown)}")
    stripped = PLACEHOLDER.sub("", text)
    digits = DIGIT.findall(stripped)
    if digits:
        ctx = [stripped[max(m.start() - 15, 0):m.end() + 15] for m in DIGIT.finditer(stripped)][:3]
        problems.append(f"raw digits found (write numbers only as placeholders): {ctx}")
    return problems


# ---------------------------------------------------------------- deterministic templates

def template(facts: dict, lang: str) -> str:
    n = facts["_n_priority"]
    if lang == "ne":
        rows = "\n".join(f"| {{{{s{i}_name}}}} | {{{{s{i}_status}}}} | {{{{s{i}_people}}}} | {{{{s{i}_nearest}}}} | "
                         f"{{{{s{i}_break_km}}}} (पुल: {{{{s{i}_break_bridge}}}}) |" for i in range(1, n + 1))
        table = ("| बस्ती | अवस्था | अनुमानित जनसंख्या | पहिलेको नजिकको सहर/अस्पताल | पहिलो अवरोधसम्म कि.मि. |\n"
                 "|---|---|---|---|---|\n" + rows) if n else "कुनै बस्तीको सडक सम्पर्क टुटेको पत्ता लागेन।"
        return f"""# स्थिति प्रतिवेदन — बाढी तथा लेदो प्रभाव (भू-उपग्रहमा आधारित)
**घटना मिति:** {{{{event_date}}}} · **क्षेत्र:** {{{{aoi}}}} · **तयार:** {{{{generated_utc}}}}

## मुख्य तथ्याङ्क
- बाढी/लेदो प्रभावित क्षेत्र: **{{{{affected_ha}}}} हेक्टर** (पानी {{{{water_ha}}}} हे., लेदो/परिवर्तित जमिन {{{{debris_ha}}}} हे.)। यसमध्ये {{{{high_conf_ha}}}} हेक्टर दुई वा बढी स्वतन्त्र स्रोतबाट पुष्टि।
- प्रभावित हुनसक्ने घर: **{{{{buildings_hit}}}}** (पानीमा {{{{buildings_in_water}}}}, लेदोमा {{{{buildings_in_debris}}}})। थप {{{{buildings_at_risk}}}} घर {{{{risk_buffer_m}}}} मिटरभित्र जोखिममा। प्रभावित घरमा अनुमानित **{{{{people_in_hit_buildings}}}} जना**।
- अवरुद्ध हुनसक्ने सडक: **{{{{road_km_blocked}}}} कि.मि.**। क्षति हुनसक्ने पुल: **{{{{bridges_total}}}} मध्ये {{{{bridges_damaged}}}}**।
- नजिकको सहर वा अस्पतालसँग सडक सम्पर्क टुटेका बस्ती: **{{{{settlements_cut_off}}}}** (अनुमानित **{{{{people_cut_off}}}} जना**)। {{{{settlements_detour}}}} बस्तीलाई लामो घुमाउरो बाटो। {{{{hospital_access_lost}}}} बस्तीको कुनै पनि अस्पतालसम्म सडक पहुँच छैन।

## प्राथमिकताका बस्ती
{table}

## विश्वसनीयता र सीमाहरू
- उपत्यकाको {{{{valley_observed_pct}}}}% भाग मात्र भू-उपग्रहले देख्न सक्यो (बाँकी रडार छाया वा बादलमा)।
- यी भू-उपग्रहमा आधारित अनुमान हुन्; जमिनमा पुष्टि गर्नुहोस्। जनसंख्या घर गणनाबाट अनुमान (प्रति घर {{{{people_per_building}}}} जना)।
- विश्लेषण क्षेत्रबाहिरका सडक खुला मानिएका छन्। भिराला पहिरोले थप सडक बन्द गरेको हुनसक्छ।

तस्बिर: पछि {{{{post_image}}}}; अघि {{{{pre_images}}}}। अप्टिकल: {{{{optical}}}}। OSM: {{{{osm_snapshot}}}}।
"""
    rows = "\n".join(f"| {{{{s{i}_name}}}} | {{{{s{i}_status}}}} | {{{{s{i}_people}}}} | {{{{s{i}_nearest}}}} | "
                     f"{{{{s{i}_break_km}}}} km (bridge: {{{{s{i}_break_bridge}}}}) |" for i in range(1, n + 1))
    table = ("| Settlement | Status | Est. people | Nearest town/hospital before | First break along route |\n"
             "|---|---|---|---|---|\n" + rows) if n else "No settlement was found to have lost its road connection."
    return f"""# Situation report — flood and debris impact (satellite-derived)
**Event date:** {{{{event_date}}}} · **Area:** {{{{aoi}}}} · **Generated:** {{{{generated_utc}}}}

## Key figures
- Flooded / debris-covered ground detected: **{{{{affected_ha}}}} ha** ({{{{water_ha}}}} ha water, {{{{debris_ha}}}} ha debris or changed ground); {{{{high_conf_ha}}}} ha confirmed by two or more independent sources.
- Buildings likely hit: **{{{{buildings_hit}}}}** ({{{{buildings_in_water}}}} in water, {{{{buildings_in_debris}}}} in debris), plus {{{{buildings_at_risk}}}} within {{{{risk_buffer_m}}}} m. About **{{{{people_in_hit_buildings}}}} people** live in the hit buildings.
- Roads likely blocked: **{{{{road_km_blocked}}}} km**. Bridges likely damaged or washed out: **{{{{bridges_damaged}}}} of {{{{bridges_total}}}}**.
- Settlements cut off by road from the nearest town or hospital: **{{{{settlements_cut_off}}}}** (about **{{{{people_cut_off}}}} people**). {{{{settlements_detour}}}} more face long detours; {{{{hospital_access_lost}}}} have no road route to any hospital.

## Priority settlements
{table}

## Confidence and limitations
- Satellites could observe {{{{valley_observed_pct}}}}% of the valley floor; the rest was in radar shadow, layover or cloud.
- These are satellite-derived estimates: confirm on the ground. Population is a proxy ({{{{people_per_building}}}} people per mapped building).
- Roads outside the analysed area are assumed open. Landslides on hillslopes may block more roads than shown.

Images: after {{{{post_image}}}}; before {{{{pre_images}}}}. Optical: {{{{optical}}}}. OSM snapshot: {{{{osm_snapshot}}}}.
"""


# ---------------------------------------------------------------- language model

SYSTEM = """You write for search-and-rescue coordinators after a flood in Nepal. Your only source of truth is the \
FACTS catalogue, produced by a satellite-mapping system. Each fact has a key, a description and its current value.

Hard rule: never type a number, digit, date or numeral of any script yourself, and never type a settlement or \
facility name yourself. Whenever you refer to a quantity, date, distance or name from the catalogue, write its \
placeholder exactly as {{key}}; the system substitutes the value afterwards. Use "-" bullets, never numbered lists. Write numbers as placeholders even inside tables and headings.

If the catalogue does not contain what is asked, say plainly that the system has no data for it. Never guess, never \
add figures from general knowledge, and keep the system's uncertainty visible (these are satellite estimates that \
need ground confirmation). Be direct and calm; no victim stories."""


def _catalogue(facts: dict) -> str:
    lines = []
    for k, v in facts.items():
        if k.startswith("_"):
            continue
        lines.append(f"{{{{{k}}}}} — {v['desc']} — current value: {v['value']}")
    return "\n".join(lines)


def _ask_anthropic(messages: list) -> tuple[str, str, object]:
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.beta.messages.create(
        model=LLM_MODEL,
        max_tokens=8000,
        system=SYSTEM,
        messages=messages,
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("model declined")
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    return text, resp.model, resp.content  # full content is echoed back unchanged on retry


def _ask_openrouter(messages: list) -> tuple[str, str, object]:
    r = requests.post(OPENROUTER_URL, timeout=180, headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}", "X-Title": "Baadhi Watch"},
        json={"model": OPENROUTER_MODEL, "max_tokens": OPENROUTER_MAX_TOKENS, "reasoning": {"effort": "low"},
              "messages": [{"role": "system", "content": SYSTEM}] + messages})
    if r.status_code != 200:
        raise RuntimeError(f"OpenRouter {r.status_code}: {r.text[:200]}")
    js = r.json()
    choice = js["choices"][0]
    if choice.get("finish_reason") in ("content_filter", "refusal"):
        raise RuntimeError("model declined")
    text = (choice["message"].get("content") or "").strip()
    if choice.get("finish_reason") == "length":
        raise RuntimeError("answer cut off by OPENROUTER_MAX_TOKENS")
    return text, f"{js.get('model', OPENROUTER_MODEL)} via OpenRouter", text


def _llm(task: str, facts: dict, lang: str) -> tuple[str, str]:
    """Returns (rendered_markdown, source). Raises on any problem so callers fall back to the template."""
    if LLM_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
        ask = _ask_anthropic
    elif LLM_PROVIDER == "openrouter" and OPENROUTER_API_KEY:
        ask = _ask_openrouter
    else:
        raise RuntimeError("no LLM key (set ANTHROPIC_API_KEY or OPENROUTER_API_KEY)")
    language = "Nepali (Devanagari script)" if lang == "ne" else "English"
    messages = [{"role": "user", "content": f"FACTS:\n{_catalogue(facts)}\n\nWrite in {language}.\n\n{task}"}]
    for _ in range(2):
        text, model, assistant_content = ask(messages)
        problems = validate(text, facts, lang)
        if not problems:
            return render(text, facts, lang), f"{model} (numbers substituted from map data)"
        messages += [{"role": "assistant", "content": assistant_content},
                     {"role": "user", "content": "Rejected by the validator: " + "; ".join(problems) +
                      ". Rewrite the whole answer using only {{key}} placeholders for every number and date."}]
    raise RuntimeError("model output failed validation twice")


def sitrep(facts: dict, lang: str = "en", use_llm: bool = True) -> dict:
    if use_llm:
        try:
            text, src = _llm("Write a one-page situation report for rescue coordinators: key figures, the priority "
                             "settlements to reach (cut off first, by estimated people), and the main limitations. "
                             "At most 350 words. Use Markdown with short sections and a table of at most eight "
                             "priority settlements.", facts, lang)
            return {"markdown": text + FOOTER[lang], "source": src}
        except Exception as ex:  # noqa: BLE001 - any failure falls back to the deterministic report
            fallback = str(ex)
    else:
        fallback = "template requested"
    return {"markdown": render(template(facts, lang), facts, lang) + FOOTER[lang], "source": f"template ({fallback})"}


def answer(facts: dict, question: str, lang: str = "en") -> dict:
    try:
        text, src = _llm("A rescuer asks the question below. Answer in at most 120 words, using only the FACTS. "
                         f"\n\nQUESTION: {question}", facts, lang)
        return {"markdown": text, "source": src}
    except Exception as ex:  # noqa: BLE001
        return {"markdown": ("The AI assistant is unavailable, so here is the system's standard summary.\n\n"
                             if lang == "en" else "एआई सहायक उपलब्ध छैन; प्रणालीको मानक सारांश:\n\n")
                + render(template(facts, lang), facts, lang), "source": f"template ({ex})"}


def save(facts: dict, path) -> None:
    path.write_text(json.dumps(facts, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
