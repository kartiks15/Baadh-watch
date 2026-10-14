import { useCallback, useEffect, useRef, useState } from 'react'
import maplibregl from 'maplibre-gl'
import { marked } from 'marked'

const API = '/api'
const STATUS_COLOR = { cut_off: '#dc2626', detour: '#f59e0b', connected: '#16a34a', no_road_pre: '#94a3b8' }
const STATUS_LABEL = { cut_off: 'Cut off', detour: 'Long detour', connected: 'Connected', no_road_pre: 'No road before' }
const RASTERS = [
  ['emsr927_agreement', 'Check vs EMSR927 (green both · red ours only · blue EMS only)'],
  ['s2_after', 'Optical after (Sentinel-2)'],
  ['s2_before', 'Optical before (Sentinel-2)'],
  ['s1_change', 'Radar change (red = darker, cyan = brighter)'],
  ['s1_after', 'Radar after'],
  ['s1_before', 'Radar before'],
  ['valley', 'Analysed valley floor'],
]
const VECTORS = [
  ['flood', 'Flood & debris'],
  ['buildings_affected', 'Buildings hit'],
  ['roads', 'Roads (pre-event)'],
  ['roads_blocked', 'Roads blocked'],
  ['bridges', 'Bridges'],
  ['settlements', 'Settlements'],
  ['targets', 'Towns & hospitals'],
  ['breaks', 'First break on route'],
]
const fmt = (v, d = 0) => (v == null ? '–' : Number(v).toLocaleString('en-US', { maximumFractionDigits: d }))

async function getJSON(url, opts) {
  const r = await fetch(url, opts)
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText)
  return r.json()
}

export default function App() {
  const mapEl = useRef(null)
  const map = useRef(null)
  const [health, setHealth] = useState(null)
  const [runs, setRuns] = useState([])
  const [bbox, setBbox] = useState([85.12, 27.88, 85.45, 28.3])
  const [date, setDate] = useState('2026-08-26')
  const [runId, setRunId] = useState(null)
  const [state, setState] = useState(null)
  const [summary, setSummary] = useState(null)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')
  const [drawing, setDrawing] = useState(false)
  const [tracing, setTracing] = useState(false)
  const [trace, setTrace] = useState(null)
  const [visible, setVisible] = useState({ s1_change: false, s2_after: true, s2_before: false, emsr927_agreement: false,
    valley: false, s1_after: false, s1_before: false,
    flood: true, buildings_affected: true, roads: true, roads_blocked: true, bridges: true, settlements: true,
    targets: true, breaks: true })
  const [opacity, setOpacity] = useState(0.8)

  // ------------------------------------------------------------------ map
  useEffect(() => {
    const m = new maplibregl.Map({
      container: mapEl.current,
      style: {
        version: 8,
        glyphs: 'https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf',
        sources: {
          base: { type: 'raster', tiles: ['https://a.tile.opentopomap.org/{z}/{x}/{y}.png', 'https://b.tile.opentopomap.org/{z}/{x}/{y}.png', 'https://c.tile.opentopomap.org/{z}/{x}/{y}.png'], tileSize: 256, maxzoom: 17,
            attribution: 'Basemap (display only): © OpenStreetMap contributors, SRTM · © OpenTopoMap (CC-BY-SA)' },
          aoi: { type: 'geojson', data: emptyFC() },
          trace: { type: 'geojson', data: emptyFC() },
        },
        layers: [
          { id: 'base', type: 'raster', source: 'base', paint: { 'raster-saturation': -0.6, 'raster-opacity': 0.85 } },
          { id: 'aoi', type: 'line', source: 'aoi', paint: { 'line-color': '#1e293b', 'line-width': 2, 'line-dasharray': [3, 2] } },
        ],
      },
      center: [85.3, 28.08], zoom: 9.3, attributionControl: { compact: true },
    })
    m.addControl(new maplibregl.NavigationControl(), 'top-right')
    m.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-right')
    map.current = m
    m.on('load', () => m.getSource('aoi').setData(bboxFC(bbox)))
    return () => m.remove()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const m = map.current
    if (m?.isStyleLoaded() && m.getSource('aoi')) m.getSource('aoi').setData(bboxFC(bbox))
  }, [bbox])

  // draw a rectangle with the mouse
  useEffect(() => {
    const m = map.current
    if (!m || !drawing) return
    m.dragPan.disable()
    m.getCanvas().style.cursor = 'crosshair'
    let start = null
    const down = (e) => { start = e.lngLat }
    const move = (e) => {
      if (!start) return
      m.getSource('aoi').setData(bboxFC(norm(start, e.lngLat)))
    }
    const up = (e) => {
      if (!start) return
      setBbox(norm(start, e.lngLat).map((v) => +v.toFixed(4)))
      start = null
      setDrawing(false)
    }
    m.on('mousedown', down); m.on('mousemove', move); m.on('mouseup', up)
    return () => {
      m.off('mousedown', down); m.off('mousemove', move); m.off('mouseup', up)
      m.dragPan.enable(); m.getCanvas().style.cursor = ''
    }
  }, [drawing])

  // ------------------------------------------------------------------ API
  const refreshRuns = useCallback(() => getJSON(`${API}/runs`).then(setRuns).catch(() => {}), [])
  useEffect(() => {
    getJSON(`${API}/health`).then(setHealth).catch((e) => setError(`API not reachable: ${e.message}`))
    refreshRuns()
  }, [refreshRuns])

  useEffect(() => {
    if (!runId) return
    let stop = false
    const poll = async () => {
      try {
        const r = await getJSON(`${API}/runs/${runId}`)
        if (stop) return
        setState(r.state)
        if (r.state.status === 'done') { setSummary(r.summary); refreshRuns(); return }
        if (r.state.status === 'error') { setError(r.state.error); return }
        setTimeout(poll, 2000)
      } catch (e) { setError(e.message) }
    }
    poll()
    return () => { stop = true }
  }, [runId, refreshRuns])

  const start = async () => {
    setError(null); setSummary(null); setTrace(null)
    try {
      const r = await getJSON(`${API}/runs`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bbox, event_date: date }) })
      setRunId(r.id)
    } catch (e) { setError(e.message) }
  }

  const openRun = (r) => { setBbox(r.bbox); setDate(r.event_date); setSummary(null); setTrace(null); setRunId(r.id) }

  // ------------------------------------------------------------------ layers
  useEffect(() => {
    const m = map.current
    if (!m || !summary) return
    const add = () => {
      addRunLayers(m, summary, visible, opacity)
      const [w, s, e, n] = summary.bbox
      m.fitBounds([[w, s], [e, n]], { padding: 40, duration: 800 })
    }
    if (m.isStyleLoaded()) add(); else m.once('load', add)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [summary])

  useEffect(() => {
    const m = map.current
    if (!m || !summary) return
    for (const [id] of [...RASTERS, ...VECTORS]) {
      for (const lid of layerIds(id)) {
        if (m.getLayer(lid)) m.setLayoutProperty(lid, 'visibility', visible[id] ? 'visible' : 'none')
      }
    }
    for (const [id] of RASTERS) if (m.getLayer(`r-${id}`)) m.setPaintProperty(`r-${id}`, 'raster-opacity', opacity)
  }, [visible, opacity, summary])

  // popups
  useEffect(() => {
    const m = map.current
    if (!m) return
    const handlers = {
      'v-settlements': (p) => `<b>${p.name}</b>${p.name_en ? ` (${p.name_en})` : ''} <i>${p.place}</i><br/>Status: <b style="color:${STATUS_COLOR[p.status]}">${STATUS_LABEL[p.status]}</b>` +
        `<br/>Est. people: ${fmt(p.est_people)} (${fmt(p.buildings_500m)} buildings ≤500 m)` +
        `<br/>Road to ${p.nearest || '–'}: ${p.pre_km ?? '–'} km before → ${p.post_km ?? 'no route'} after` +
        (p.break_km != null ? `<br/>First break ${p.break_km} km along route${p.break_is_bridge === true || p.break_is_bridge === 'true' ? ' (bridge)' : ''}` : ''),
      'v-bridges': (p) => `<b>Bridge</b> ${p.name || ''}<br/>${p.status === 'likely_damaged' ? '<b style="color:#dc2626">Likely damaged / washed out</b>' : 'No flood detected nearby'}<br/>${p.highway}`,
      'v-buildings_affected': (p) => `<b>Building</b><br/>${p.status.replace('_', ' ')} (confidence ${p.confidence})`,
      'v-flood': (p) => `<b>${p.class === 'water' ? 'Flood water' : 'Debris / changed ground'}</b><br/>${p.area_ha} ha · confidence ${p.confidence}<br/>mean evidence votes ${p.mean_votes}`,
      'v-roads_blocked': (p) => `<b>Road likely blocked</b> ${p.name || ''}<br/>${p.highway}, ${fmt(p.length_m)} m${p.bridge ? ' (bridge)' : ''}`,
      'v-targets': (p) => `<b>${p.name}</b><br/>${p.kind}`,
    }
    const fns = Object.entries(handlers).map(([lid, html]) => {
      const fn = (e) => {
        if (tracing || drawing) return
        new maplibregl.Popup({ maxWidth: '300px' }).setLngLat(e.lngLat).setHTML(html(e.features[0].properties)).addTo(m)
      }
      m.on('click', lid, fn)
      return [lid, fn]
    })
    return () => fns.forEach(([lid, fn]) => m.off('click', lid, fn))
  }, [tracing, drawing])

  // flow-path trace (bonus)
  useEffect(() => {
    const m = map.current
    if (!m || !tracing || !runId) return
    m.getCanvas().style.cursor = 'crosshair'
    const click = async (e) => {
      setTracing(false)
      try {
        const r = await getJSON(`${API}/runs/${runId}/trace`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ lon: e.lngLat.lng, lat: e.lngLat.lat }) })
        setTrace(r)
        const fc = { type: 'FeatureCollection', features: [r.path, ...r.settlements.map((s) => ({ type: 'Feature',
          geometry: { type: 'Point', coordinates: [s.lon, s.lat] }, properties: s }))] }
        m.getSource('trace').setData(fc)
        setTab('trace')
      } catch (err) { setError(err.message) }
    }
    m.once('click', click)
    return () => { m.off('click', click); m.getCanvas().style.cursor = '' }
  }, [tracing, runId])

  const flyTo = (lon, lat) => map.current?.flyTo({ center: [lon, lat], zoom: 13 })
  const running = state && ['queued', 'running'].includes(state.status)

  return (
    <div className="app">
      <header>
        <div className="brand">
          <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden="true"><path d="M2 20c4-4 8 4 12 0s8 4 12 0 4-2 4-2v10H2z" fill="var(--water)" /><circle cx="24" cy="8" r="4" fill="var(--warn)" /></svg>
          <div><h1>Baadhi Watch</h1><span>Flood damage &amp; cut-off settlements from free satellite data</span></div>
        </div>
        <div className="chips">
          <Chip ok={health?.sentinel_hub} label="Sentinel Hub" />
          <Chip ok={health?.llm} label="AI copilot" />
          <Chip ok={health?.model?.startsWith('U-Net')} label={health?.model?.startsWith('U-Net') ? `U-Net IoU ${health.model_metrics?.bolivia?.iou_water ?? ''} (unseen)` : 'model'} />
        </div>
      </header>

      <aside className="left panel">
        <section>
          <h2>1 · Area and flood date</h2>
          <div className="bbox">
            {['W', 'S', 'E', 'N'].map((k, i) => (
              <label key={k}>{k}<input type="number" step="0.01" value={bbox[i]} onChange={(e) => {
                const b = [...bbox]; b[i] = +e.target.value; setBbox(b) }} /></label>
            ))}
          </div>
          <div className="row">
            <button className={drawing ? 'active' : ''} onClick={() => setDrawing(!drawing)}>{drawing ? 'Drag on map…' : 'Draw area'}</button>
            <button onClick={() => { const c = health?.case_study; if (c) { setBbox(c.bbox); setDate(c.event_date) } }}>Trishuli case study</button>
          </div>
          <label className="date">Flood date<input type="date" value={date} onChange={(e) => setDate(e.target.value)} /></label>
          <button className="primary" disabled={running} onClick={start}>{running ? `Running… ${state.progress}%` : 'Run analysis'}</button>
          {error && <p className="error">{error}</p>}
          {state && (
            <details className="log" open={running}>
              <summary>Pipeline log ({state.status}{state.seconds ? `, ${state.seconds}s` : ''})</summary>
              <pre>{state.log.slice(-12).join('\n')}</pre>
            </details>
          )}
        </section>
        <section>
          <h2>Layers</h2>
          {[...VECTORS, ...RASTERS].map(([id, label]) => (
            <label key={id} className="check"><input type="checkbox" checked={visible[id]} onChange={() => setVisible({ ...visible, [id]: !visible[id] })} />{label}</label>
          ))}
          <label className="range">Imagery opacity<input type="range" min="0" max="1" step="0.05" value={opacity} onChange={(e) => setOpacity(+e.target.value)} /></label>
          <Legend />
        </section>
        <section>
          <h2>Previous runs</h2>
          {runs.length === 0 && <p className="muted">None yet.</p>}
          {runs.slice(0, 8).map((r) => (
            <button key={r.id} className="runrow" onClick={() => openRun(r)}>
              <span>{r.event_date}</span><span className="muted">{r.bbox.map((v) => v.toFixed(2)).join(', ')}</span><span className={`st ${r.status}`}>{r.status}</span>
            </button>
          ))}
        </section>
      </aside>

      {/* MapLibre adds its own classes to the container; keep React-managed classes on a wrapper instead */}
      <main className={`map ${drawing || tracing ? 'picking' : ''}`}><div ref={mapEl} className="map-canvas" /></main>

      <aside className="right panel">
        <nav className="tabs">
          {[['overview', 'Overview'], ['sitrep', 'Sitrep'], ['ask', 'Ask'], ['trace', 'Flow path'], ['model', 'Model']].map(([k, l]) => (
            <button key={k} className={tab === k ? 'active' : ''} onClick={() => setTab(k)}>{l}</button>
          ))}
        </nav>
        {!summary && tab !== 'model' && <Empty running={running} />}
        {summary && tab === 'overview' && <Overview s={summary} runId={runId} flyTo={flyTo} />}
        {summary && tab === 'sitrep' && <Sitrep runId={runId} llm={health?.llm} />}
        {summary && tab === 'ask' && <Ask runId={runId} />}
        {summary && tab === 'trace' && <Trace trace={trace} tracing={tracing} setTracing={setTracing} flyTo={flyTo} />}
        {tab === 'model' && <Model health={health} runId={summary ? runId : null} />}
      </aside>

      <footer>Contains modified Copernicus Sentinel data 2026 · Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA · © OpenStreetMap contributors · Educational prototype, not an operational tool.</footer>
    </div>
  )
}

function Chip({ ok, label }) {
  return <span className={`chip ${ok ? 'ok' : 'off'}`}><i />{label}</span>
}

function Empty({ running }) {
  return (
    <div className="empty">
      <p><b>{running ? 'Analysis running…' : 'No analysis loaded.'}</b></p>
      <p className="muted">Pick an area and a flood date, then run. The system finds Sentinel-1 radar images from the same orbit track before and after the date, maps new water and debris on the valley floor, overlays pre-event OpenStreetMap, and works out which settlements lost their road link to the nearest town or hospital.</p>
    </div>
  )
}

function Overview({ s, runId, flyTo }) {
  const c = s.network.counts
  const d = s.detect
  const b = s.buildings
  const km = Object.values(s.roads.km_by_class).reduce((a, v) => a + v, 0)
  const pri = s.network.settlements.filter((r) => r.status === 'cut_off' || r.status === 'detour')
  const file = (n) => `${API}/runs/${runId}/files/${n}`
  return (
    <div className="scroll">
      <div className="kpis">
        <Kpi v={fmt(c.cut_off)} l="settlements cut off" sub={`≈ ${fmt(c.people_cut_off)} people`} tone="bad" />
        <Kpi v={fmt(b.in_water + b.in_debris)} l="buildings hit" sub={`${fmt(b.at_risk)} more at risk`} tone="bad" />
        <Kpi v={fmt(d.affected_ha, 0)} l="ha flooded / debris" sub={`${fmt(d.high_conf_ha, 0)} ha high-confidence`} />
        <Kpi v={fmt(km, 1)} l="km road blocked" sub={`${fmt(s.bridges.likely_damaged)} of ${fmt(s.bridges.total)} bridges hit`} />
        <Kpi v={fmt(c.detour)} l="long detours" sub={`${fmt(c.hospital_access_lost)} lost hospital access`} tone="warn" />
        <Kpi v={`${fmt(d.valley_observed_pct, 0)}%`} l="valley floor observed" sub="rest: radar shadow / cloud" />
      </div>
      <h3>Priority settlements</h3>
      {pri.length === 0 ? <p className="muted">No settlement lost its road connection.</p> : (
        <table className="pri">
          <thead><tr><th>Settlement</th><th>Status</th><th>≈ people</th><th>Break</th></tr></thead>
          <tbody>
            {pri.slice(0, 40).map((r) => (
              <tr key={r.osm_id || r.name} onClick={() => flyTo(r.lon, r.lat)}>
                <td title={r.name}>{r.name_en || r.name}</td>
                <td><span className="dot" style={{ background: STATUS_COLOR[r.status] }} />{STATUS_LABEL[r.status]}</td>
                <td>{fmt(r.est_people)}</td>
                <td>{r.cause ? `${r.cause.km_along_route} km${r.cause.bridge ? ' · bridge' : ''}` : '–'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <h3>Inputs used</h3>
      <ul className="inputs">
        {s.pairs.map((p) => <li key={p.post.label}>After: {p.post.label}<br /><span className="muted">Before: {p.pres.map((x) => x.label).join('; ')}</span></li>)}
        <li>Optical: {s.s2_note}</li>
        <li>OSM: {s.osm_snapshot} ({s.osm_source})</li>
        <li>Water model: {s.model}</li>
      </ul>
      <h3>Exports for field teams</h3>
      <div className="exports">
        <a href={file('priority.csv')} download>Priority list (CSV)</a>
        <a href={file('cutoff.kml')} download>Cut-off villages (KML)</a>
        <a href={file('flood.geojson')} download>Flood extent (GeoJSON)</a>
        <a href={file('settlements.geojson')} download>Settlements (GeoJSON)</a>
        <a href={file('roads_blocked.geojson')} download>Blocked roads (GeoJSON)</a>
      </div>
    </div>
  )
}

function Kpi({ v, l, sub, tone }) {
  return <div className={`kpi ${tone || ''}`}><b>{v}</b><span>{l}</span><small>{sub}</small></div>
}

function Sitrep({ runId, llm }) {
  const [lang, setLang] = useState('en')
  const [ai, setAi] = useState(false)
  const [doc, setDoc] = useState(null)
  const [busy, setBusy] = useState(false)
  useEffect(() => {
    setBusy(true); setDoc(null)
    getJSON(`${API}/runs/${runId}/sitrep?lang=${lang}&ai=${ai}`).then(setDoc).catch((e) => setDoc({ markdown: e.message, source: 'error' })).finally(() => setBusy(false))
  }, [runId, lang, ai])
  const print = () => {
    const w = window.open('', '_blank')
    w.document.write(`<html><head><meta charset="utf-8"><title>Situation report</title><style>body{font:13px/1.45 system-ui,sans-serif;max-width:780px;margin:24px auto;color:#111}table{border-collapse:collapse;width:100%;font-size:12px}td,th{border:1px solid #ccc;padding:3px 6px;text-align:left}h1{font-size:20px}h2{font-size:15px;margin-top:14px}</style></head><body>${marked.parse(doc.markdown)}</body></html>`)
    w.document.close(); w.focus(); w.print()
  }
  return (
    <div className="scroll">
      <div className="row seg">
        <button className={lang === 'en' ? 'active' : ''} onClick={() => setLang('en')}>English</button>
        <button className={lang === 'ne' ? 'active' : ''} onClick={() => setLang('ne')}>नेपाली</button>
        <span className="sp" />
        <button className={!ai ? 'active' : ''} onClick={() => setAi(false)}>Template</button>
        <button className={ai ? 'active' : ''} disabled={!llm} title={llm ? '' : 'Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY in .env to enable'} onClick={() => setAi(true)}>AI-written</button>
      </div>
      {busy && <p className="muted">{ai ? 'Writing report (every number is filled in from the map data)…' : 'Loading…'}</p>}
      {doc && <>
        <article className="md" dangerouslySetInnerHTML={{ __html: marked.parse(doc.markdown) }} />
        <p className="muted small">Source: {doc.source}</p>
        <button onClick={print}>Print / save as PDF</button>
      </>}
    </div>
  )
}

function Ask({ runId }) {
  const [q, setQ] = useState('')
  const [lang, setLang] = useState('en')
  const [items, setItems] = useState([])
  const [busy, setBusy] = useState(false)
  const send = async (e) => {
    e.preventDefault()
    if (!q.trim()) return
    const question = q
    setQ(''); setBusy(true)
    try {
      const r = await getJSON(`${API}/runs/${runId}/ask`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, lang }) })
      setItems((it) => [...it, { q: question, ...r }])
    } catch (err) { setItems((it) => [...it, { q: question, markdown: err.message, source: 'error' }]) }
    setBusy(false)
  }
  const examples = ['Which villages should helicopters reach first?', 'How many bridges are damaged?', 'कति बस्ती सम्पर्कविहीन छन्?']
  return (
    <div className="scroll chat">
      <p className="muted small">Answers use only this run's numbers. The assistant writes placeholders and the system fills in each value, so it cannot invent figures.</p>
      <div className="row seg">
        <button className={lang === 'en' ? 'active' : ''} onClick={() => setLang('en')}>English</button>
        <button className={lang === 'ne' ? 'active' : ''} onClick={() => setLang('ne')}>नेपाली</button>
      </div>
      <div className="examples">{examples.map((x) => <button key={x} onClick={() => setQ(x)}>{x}</button>)}</div>
      {items.map((it, i) => (
        <div key={i} className="qa"><p className="q">{it.q}</p><div className="md" dangerouslySetInnerHTML={{ __html: marked.parse(it.markdown) }} /><p className="muted small">{it.source}</p></div>
      ))}
      {busy && <p className="muted">Thinking…</p>}
      <form onSubmit={send} className="ask"><input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask about this flood…" /><button className="primary" disabled={busy}>Ask</button></form>
    </div>
  )
}

function Trace({ trace, tracing, setTracing, flyTo }) {
  return (
    <div className="scroll">
      <p className="muted small">Click any point upstream. The system follows the steepest-descent flow path on the Copernicus DEM and lists settlements on the valley floor along it, in the order a flood would reach them.</p>
      <button className={`primary ${tracing ? 'active' : ''}`} onClick={() => setTracing(!tracing)}>{tracing ? 'Click a point on the map…' : 'Pick upstream point'}</button>
      {trace && <>
        <p><b>{trace.path.properties.length_km} km</b> traced, {fmt(trace.path.properties.drop_m)} m drop. {trace.settlements.length} settlements along the corridor, in the order the water would reach them; {fmt(trace.buildings_near_river)} buildings sit ≤25 m above river level within 1.5 km of the path.</p>
        <table className="pri">
          <thead><tr><th>km</th><th>Settlement</th><th>m above river</th><th>Houses near river</th><th>Arrival*</th></tr></thead>
          <tbody>{trace.settlements.map((s) => (
            <tr key={s.name + s.km_downstream} onClick={() => flyTo(s.lon, s.lat)}>
              <td>{s.km_downstream}</td><td title={s.name}>{s.name_en || s.name}</td><td><span className="dot" style={{ background: { high: '#dc2626', medium: '#f59e0b', low: '#16a34a' }[s.risk] || '#94a3b8' }} />{s.height_above_river_m ?? '–'}</td><td>{fmt(s.buildings_near_river)}</td><td>{s.eta_min[0]}–{s.eta_min[1]} min</td>
            </tr>))}</tbody>
        </table>
        <p className="muted small">* {trace.note}</p>
      </>}
    </div>
  )
}

function Model({ health, runId }) {
  const [emsr, setEmsr] = useState(null)
  useEffect(() => {
    if (!runId) return
    getJSON(`${API}/runs/${runId}/files/emsr927_eval.json`).then(setEmsr).catch(() => setEmsr(null))
  }, [runId])
  const m = health?.model_metrics
  return (
    <div className="scroll">
      <h3>Water segmentation model</h3>
      <p className="small">{health?.model}. U-Net (ResNet-34 encoder) on Sentinel-1 VV/VH, trained on Sen1Floods11 hand-labelled chips (Bonafilia et al., 2020). Applied to the post-event and pre-event images; "new water" = water after but not before.</p>
      {m ? (
        <table className="pri">
          <thead><tr><th>Split (never trained on)</th><th>IoU</th><th>F1</th><th>Precision</th><th>Recall</th></tr></thead>
          <tbody>
            {[['valid', 'Validation'], ['test', 'Test (held-out chips)'], ['bolivia', 'Bolivia (held-out event)']].map(([k, l]) => m[k] && (
              <tr key={k}><td>{l}</td><td>{m[k].iou_water}</td><td>{m[k].f1_water}</td><td>{m[k].precision}</td><td>{m[k].recall}</td></tr>
            ))}
          </tbody>
        </table>
      ) : <p className="muted">No model metrics found.</p>}
      <h3>Case study vs Copernicus EMS (EMSR927)</h3>
      {emsr ? <pre className="small">{JSON.stringify(emsr.summary || emsr, null, 1)}</pre> :
        <p className="muted small">Run <code>python -m eval.evaluate_emsr --run &lt;id&gt;</code> after placing the EMSR927 vectors in <code>validation/</code>. EMS maps are used only for checking, never as inputs.</p>}
    </div>
  )
}

function Legend() {
  return (
    <div className="legend">
      <span><i style={{ background: 'var(--water)' }} />Flood water</span>
      <span><i style={{ background: 'var(--debris)' }} />Debris / changed ground</span>
      <span><i className="line" style={{ background: '#dc2626' }} />Road blocked</span>
      {Object.entries(STATUS_LABEL).map(([k, l]) => <span key={k}><i className="round" style={{ background: STATUS_COLOR[k] }} />{l}</span>)}
    </div>
  )
}

// ------------------------------------------------------------------ helpers
function emptyFC() { return { type: 'FeatureCollection', features: [] } }
function norm(a, b) { return [Math.min(a.lng, b.lng), Math.min(a.lat, b.lat), Math.max(a.lng, b.lng), Math.max(a.lat, b.lat)] }
function bboxFC([w, s, e, n]) {
  return { type: 'FeatureCollection', features: [{ type: 'Feature', properties: {}, geometry: { type: 'Polygon', coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] } }] }
}
function layerIds(id) {
  if (RASTERS.some(([r]) => r === id)) return [`r-${id}`]
  if (id === 'settlements') return ['v-settlements', 'v-settlements-label']
  if (id === 'targets') return ['v-targets', 'v-targets-label']
  if (id === 'flood') return ['v-flood', 'v-flood-line']
  return [`v-${id}`]
}
function addRunLayers(m, s, visible, opacity) {
  const base = `${API}/runs/${s.id}/files/`
  for (const lid of m.getStyle().layers.map((l) => l.id)) if (lid.startsWith('r-') || lid.startsWith('v-') || lid.startsWith('t-')) m.removeLayer(lid)
  for (const sid of Object.keys(m.getStyle().sources)) if (sid.startsWith('src-')) m.removeSource(sid)
  const vis = (id) => (visible[id] ? 'visible' : 'none')
  for (const [id] of [...RASTERS].reverse()) {
    m.addSource(`src-${id}`, { type: 'image', url: base + `${id}.png`, coordinates: s.grid.corners })
    m.addLayer({ id: `r-${id}`, type: 'raster', source: `src-${id}`, layout: { visibility: vis(id) }, paint: { 'raster-opacity': opacity, 'raster-fade-duration': 0 } })
  }
  const gj = (id) => m.addSource(`src-${id}`, { type: 'geojson', data: base + `${id}.geojson` })
  ;['roads', 'flood', 'buildings_affected', 'roads_blocked', 'bridges', 'breaks', 'targets', 'settlements'].forEach(gj)
  m.addLayer({ id: 'v-roads', type: 'line', source: 'src-roads', layout: { visibility: vis('roads') }, paint: { 'line-color': '#475569', 'line-width': ['interpolate', ['linear'], ['zoom'], 9, 0.5, 14, 2], 'line-opacity': 0.7 } })
  m.addLayer({ id: 'v-flood', type: 'fill', source: 'src-flood', layout: { visibility: vis('flood') }, paint: {
    'fill-color': ['match', ['get', 'class'], 'water', '#2563eb', '#b45309'],
    'fill-opacity': ['match', ['get', 'confidence'], 'high', 0.7, 'medium', 0.5, 0.3] } })
  m.addLayer({ id: 'v-flood-line', type: 'line', source: 'src-flood', layout: { visibility: vis('flood') }, paint: { 'line-color': ['match', ['get', 'class'], 'water', '#1d4ed8', '#92400e'], 'line-width': 0.8 } })
  m.addLayer({ id: 'v-buildings_affected', type: 'fill', source: 'src-buildings_affected', layout: { visibility: vis('buildings_affected') }, paint: {
    'fill-color': ['match', ['get', 'status'], 'in_water', '#1e3a8a', 'in_debris', '#7c2d12', '#f97316'], 'fill-opacity': 0.9 } })
  m.addLayer({ id: 'v-roads_blocked', type: 'line', source: 'src-roads_blocked', layout: { visibility: vis('roads_blocked') }, paint: { 'line-color': '#dc2626', 'line-width': ['interpolate', ['linear'], ['zoom'], 9, 2.5, 14, 6] } })
  m.addLayer({ id: 'v-bridges', type: 'line', source: 'src-bridges', layout: { visibility: vis('bridges') }, paint: {
    'line-color': ['match', ['get', 'status'], 'likely_damaged', '#dc2626', '#111827'], 'line-width': 6, 'line-dasharray': [1, 0.6] } })
  m.addLayer({ id: 'v-breaks', type: 'circle', source: 'src-breaks', layout: { visibility: vis('breaks') }, paint: { 'circle-radius': 5, 'circle-color': '#fff', 'circle-stroke-color': '#dc2626', 'circle-stroke-width': 3 } })
  m.addLayer({ id: 'v-targets', type: 'circle', source: 'src-targets', layout: { visibility: vis('targets') }, paint: { 'circle-radius': 7, 'circle-color': ['match', ['get', 'kind'], 'hospital', '#be123c', '#0f172a'], 'circle-stroke-color': '#fff', 'circle-stroke-width': 2 } })
  m.addLayer({ id: 'v-targets-label', type: 'symbol', source: 'src-targets', layout: { visibility: vis('targets'), 'text-field': ['concat', ['match', ['get', 'kind'], 'hospital', '✚ ', ''], ['get', 'name']], 'text-size': 11, 'text-offset': [0, 1.2], 'text-anchor': 'top', 'text-font': ['Noto Sans Regular'] }, paint: { 'text-color': '#0f172a', 'text-halo-color': '#fff', 'text-halo-width': 1.5 } })
  m.addLayer({ id: 'v-settlements', type: 'circle', source: 'src-settlements', layout: { visibility: vis('settlements') }, paint: {
    'circle-radius': ['interpolate', ['linear'], ['get', 'est_people'], 0, 4, 500, 7, 3000, 12],
    'circle-color': ['match', ['get', 'status'], 'cut_off', STATUS_COLOR.cut_off, 'detour', STATUS_COLOR.detour, 'connected', STATUS_COLOR.connected, STATUS_COLOR.no_road_pre],
    'circle-stroke-color': '#fff', 'circle-stroke-width': 1.5 } })
  m.addLayer({ id: 'v-settlements-label', type: 'symbol', source: 'src-settlements', minzoom: 10, layout: { visibility: vis('settlements'), 'text-field': ['coalesce', ['get', 'name_en'], ['get', 'name']], 'text-size': 11, 'text-offset': [0, 1], 'text-anchor': 'top', 'text-font': ['Noto Sans Regular'] }, paint: { 'text-color': '#111827', 'text-halo-color': '#fff', 'text-halo-width': 1.2 } })
  if (!m.getLayer('t-trace')) {
    m.addLayer({ id: 't-trace', type: 'line', source: 'trace', filter: ['==', '$type', 'LineString'], paint: { 'line-color': '#7c3aed', 'line-width': 4 } })
    m.addLayer({ id: 't-trace-pts', type: 'circle', source: 'trace', filter: ['==', '$type', 'Point'], paint: { 'circle-radius': 6, 'circle-color': '#7c3aed', 'circle-stroke-color': '#fff', 'circle-stroke-width': 2 } })
  }
}
