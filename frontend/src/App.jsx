import { useCallback, useEffect, useRef, useState } from 'react'
import maplibregl from 'maplibre-gl'
import { marked } from 'marked'

const API = '/api'
const STATUS_COLOR = { cut_off: '#ef4444', detour: '#f59e0b', connected: '#22c55e', no_road_pre: '#94a3b8' }
const STATUS_LABEL = { cut_off: 'Cut off', detour: 'Long detour', connected: 'Connected', no_road_pre: 'No road before' }
const RASTERS = [
  ['emsr927_agreement', 'Agreement with EMSR927'],
  ['s2_after', 'Optical after'],
  ['s2_before', 'Optical before'],
  ['s1_change', 'Radar change'],
  ['s1_after', 'Radar after'],
  ['s1_before', 'Radar before'],
  ['valley', 'Flood-reachable zone'],
]
const VECTORS = [
  ['flood', 'Flood & debris'],
  ['buildings_affected', 'Buildings hit'],
  ['roads_blocked', 'Blocked roads'],
  ['bridges', 'Bridges'],
  ['settlements', 'Villages'],
  ['targets', 'Towns & hospitals'],
  ['breaks', 'Road break points'],
  ['roads', 'Road network'],
]
const LAYER_GROUPS = [
  ['Impact', VECTORS.map(([k]) => k)],
  ['Satellite imagery', ['s2_after', 's2_before', 's1_change', 's1_after', 's1_before', 'valley']],
  ['Validation', ['emsr927_agreement']],
]
const LABEL = Object.fromEntries([...RASTERS, ...VECTORS])
const STEPS = [
  ['scenes', 'Finding satellite passes'], ['osm', 'Pre-flood map data'], ['terrain', 'Terrain & flood zone'],
  ['radar', 'Radar images'], ['optical', 'Optical images'], ['detect', 'Detecting flood & debris'],
  ['impact', 'Buildings & bridges'], ['network', 'Roads & cut-off villages'], ['report', 'Situation report'],
]
const ACTIVE = ['queued', 'running']
const DEFAULT_CASE = { bbox: [84.93, 27.78, 85.45, 28.3], event_date: '2026-08-26' }
const TABS = [['overview', 'Overview'], ['report', 'Report'], ['ask', 'Ask'], ['trace', 'Flow path'], ['accuracy', 'Accuracy']]
const fmt = (v, d = 0) => (v == null ? '–' : Number(v).toLocaleString('en-US', { maximumFractionDigits: d }))

async function getJSON(url, opts) {
  const r = await fetch(url, opts)
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText)
  return r.json()
}
const postJSON = (url, body) => getJSON(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

export default function App() {
  const mapEl = useRef(null)
  const map = useRef(null)
  const [health, setHealth] = useState(null)
  const [runs, setRuns] = useState([])
  const [bbox, setBbox] = useState([84.93, 27.78, 85.45, 28.3])
  const [date, setDate] = useState('2026-08-26')
  const [runId, setRunId] = useState(null)
  const [state, setState] = useState(null)
  const [summary, setSummary] = useState(null)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')
  const [drawing, setDrawing] = useState(false)
  const [tracing, setTracing] = useState(false)
  const [trace, setTrace] = useState(null)
  const [visible, setVisible] = useState({ s2_after: false, s2_before: false, s1_change: false, s1_after: false,
    s1_before: false, valley: false, emsr927_agreement: false, flood: true, buildings_affected: true, roads: false,
    roads_blocked: true, bridges: true, settlements: true, targets: true, breaks: true })
  const [opacity, setOpacity] = useState(0.75)
  const [cardOpen, setCardOpen] = useState(true)
  const [drawerOpen, setDrawerOpen] = useState(true)
  const [panel, setPanel] = useState(null) // 'layers' | 'history' | 'system'

  const running = state && ACTIVE.includes(state.status)
  const caseStudy = health?.case_study || DEFAULT_CASE

  // ------------------------------------------------------------------ map
  useEffect(() => {
    const m = new maplibregl.Map({
      container: mapEl.current,
      style: {
        version: 8,
        glyphs: 'https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf',
        sources: {
          base: { type: 'raster', tiles: ['a', 'b', 'c'].map((s) => `https://${s}.tile.opentopomap.org/{z}/{x}/{y}.png`),
            tileSize: 256, maxzoom: 17, attribution: 'Basemap (display only): © OpenStreetMap contributors, SRTM · © OpenTopoMap (CC-BY-SA)' },
          aoi: { type: 'geojson', data: emptyFC() },
          trace: { type: 'geojson', data: emptyFC() },
        },
        layers: [
          { id: 'base', type: 'raster', source: 'base', paint: { 'raster-saturation': -0.45, 'raster-contrast': -0.08 } },
          { id: 'aoi-fill', type: 'fill', source: 'aoi', paint: { 'fill-color': '#4f8cff', 'fill-opacity': 0.06 } },
          { id: 'aoi', type: 'line', source: 'aoi', paint: { 'line-color': '#2f6bff', 'line-width': 2, 'line-dasharray': [3, 2] } },
        ],
      },
      center: [85.2, 28.04], zoom: 9.2, attributionControl: { compact: true },
    })
    m.addControl(new maplibregl.NavigationControl({ visualizePitch: false }), 'bottom-right')
    m.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-right')
    map.current = m
    if (new URLSearchParams(window.location.search).has('debug')) window.__bwMap = m
    m.on('load', () => {
      m.getSource('aoi').setData(bboxFC(bbox))
      const [w, s, e, n] = bbox
      m.fitBounds([[w, s], [e, n]], { padding: fitPadding(m, 360, 440), duration: 0 })
      document.querySelector('.maplibregl-ctrl-attrib')?.classList.remove('maplibregl-compact-show')
    })
    // safety net: if the map was created while hidden (0×0), resize it once the container has a real size
    const fit = () => {
      const c = m.getCanvas(), el = mapEl.current
      if (el && (Math.abs(c.clientWidth - el.clientWidth) > 1 || Math.abs(c.clientHeight - el.clientHeight) > 1)) m.resize()
    }
    window.addEventListener('resize', fit)
    const t = setInterval(fit, 1000)
    return () => { window.removeEventListener('resize', fit); clearInterval(t); m.remove() }
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
    let start = null
    const down = (e) => { start = e.lngLat }
    const move = (e) => { if (start) m.getSource('aoi').setData(bboxFC(norm(start, e.lngLat))) }
    const up = (e) => {
      if (!start) return
      setBbox(norm(start, e.lngLat).map((v) => +v.toFixed(4)))
      start = null
      setDrawing(false)
    }
    m.on('mousedown', down); m.on('mousemove', move); m.on('mouseup', up)
    return () => { m.off('mousedown', down); m.off('mousemove', move); m.off('mouseup', up); m.dragPan.enable() }
  }, [drawing])

  // ------------------------------------------------------------------ API
  const refreshRuns = useCallback(() => getJSON(`${API}/runs`).then(setRuns).catch(() => {}), [])
  useEffect(() => {
    getJSON(`${API}/health`).then(setHealth).catch((e) => setError(`The API is not reachable: ${e.message}`))
    refreshRuns()
  }, [refreshRuns])

  const anyActive = runs.some((r) => ACTIVE.includes(r.status))
  useEffect(() => {
    if (!anyActive && panel !== 'history') return
    const t = setInterval(refreshRuns, 4000)
    return () => clearInterval(t)
  }, [anyActive, panel, refreshRuns])

  useEffect(() => {
    if (!runId) return
    let stop = false
    const poll = async () => {
      try {
        const r = await getJSON(`${API}/runs/${runId}`)
        if (stop) return
        setState(r.state)
        if (r.state.status === 'done') {
          setSummary(r.summary); setCardOpen(false); setDrawerOpen(true); refreshRuns(); return
        }
        if (!ACTIVE.includes(r.state.status)) { setError(r.state.error || `Run ${r.state.status}.`); refreshRuns(); return }
        setTimeout(poll, 2000)
      } catch (e) { setError(e.message) }
    }
    poll()
    return () => { stop = true }
  }, [runId, refreshRuns])

  const start = async (b = bbox, d = date) => {
    setError(null); setSummary(null); setTrace(null); setTab('overview')
    try {
      const r = await postJSON(`${API}/runs`, { bbox: b, event_date: d })
      setRunId(r.id); setCardOpen(true); refreshRuns()
    } catch (e) { setError(e.message) }
  }
  const openCaseStudy = () => {
    if (!caseStudy) return
    setBbox(caseStudy.bbox); setDate(caseStudy.event_date); start(caseStudy.bbox, caseStudy.event_date)
  }
  const openRun = (r) => {
    setPanel(null); setError(null); setBbox(r.bbox); setDate(r.event_date); setSummary(null); setTrace(null)
    setTab('overview'); setRunId(null); setTimeout(() => setRunId(r.id), 0)
  }
  const cancelRun = async (id) => { await postJSON(`${API}/runs/${id}/cancel`, {}).catch((e) => setError(e.message)); refreshRuns() }
  const deleteRun = async (id) => {
    await getJSON(`${API}/runs/${id}`, { method: 'DELETE' }).catch((e) => setError(e.message))
    if (id === runId) { setRunId(null); setState(null); setSummary(null); clearRunLayers(map.current) }
    refreshRuns()
  }

  // ------------------------------------------------------------------ layers
  useEffect(() => {
    const m = map.current
    if (!m || !summary) return
    const add = () => {
      addRunLayers(m, summary, visible, opacity)
      const [w, s, e, n] = summary.bbox
      m.fitBounds([[w, s], [e, n]], { padding: fitPadding(m, 40, 460), duration: 900 })
    }
    if (m.isStyleLoaded()) add(); else m.once('load', add)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [summary])

  useEffect(() => {
    const m = map.current
    if (!m || !summary) return
    for (const id of Object.keys(LABEL)) {
      for (const lid of layerIds(id)) if (m.getLayer(lid)) m.setLayoutProperty(lid, 'visibility', visible[id] ? 'visible' : 'none')
    }
    for (const [id] of RASTERS) if (m.getLayer(`r-${id}`)) m.setPaintProperty(`r-${id}`, 'raster-opacity', opacity)
  }, [visible, opacity, summary])

  // popups
  useEffect(() => {
    const m = map.current
    if (!m) return
    const handlers = {
      'v-settlements': (p) => `<div class="pop-title">${p.name_en || p.name}</div>` +
        `${p.name_en && p.name !== p.name_en ? `<div class="pop-sub">${p.name} · ${p.place}</div>` : `<div class="pop-sub">${p.place}</div>`}` +
        `<div class="pop-status" style="--c:${STATUS_COLOR[p.status]}">${STATUS_LABEL[p.status]}</div>` +
        `<div class="pop-row"><span>Estimated people</span><b>${fmt(p.est_people)}</b></div>` +
        `<div class="pop-row"><span>Road to ${p.nearest || 'nearest town'}</span><b>${p.pre_km ?? '–'} km → ${p.post_km != null ? `${p.post_km} km` : 'no route'}</b></div>` +
        (p.break_km != null ? `<div class="pop-row"><span>First break on route</span><b>${p.break_km} km${String(p.break_is_bridge) === 'true' ? ' · bridge' : ''}</b></div>` : ''),
      'v-bridges': (p) => `<div class="pop-title">Bridge ${p.name || ''}</div><div class="pop-status" style="--c:${p.status === 'likely_damaged' ? '#ef4444' : '#22c55e'}">${p.status === 'likely_damaged' ? 'Likely damaged / washed out' : 'No flood detected nearby'}</div><div class="pop-sub">${p.highway}</div>`,
      'v-buildings_affected': (p) => `<div class="pop-title">Building</div><div class="pop-status" style="--c:${p.status === 'at_risk' ? '#f59e0b' : '#ef4444'}">${p.status.replace('_', ' ')}</div><div class="pop-sub">confidence ${p.confidence}</div>`,
      'v-flood': (p) => `<div class="pop-title">${p.class === 'water' ? 'Flood water' : 'Debris / changed ground'}</div><div class="pop-row"><span>Area</span><b>${p.area_ha} ha</b></div><div class="pop-row"><span>Confidence</span><b>${p.confidence}</b></div>`,
      'v-roads_blocked': (p) => `<div class="pop-title">Road likely blocked</div><div class="pop-sub">${p.name || p.highway}</div><div class="pop-row"><span>Length</span><b>${fmt(p.length_m)} m${p.bridge ? ' · bridge' : ''}</b></div>`,
      'v-targets': (p) => `<div class="pop-title">${p.name}</div><div class="pop-sub">${p.kind}</div>`,
    }
    const fns = Object.entries(handlers).map(([lid, html]) => {
      const fn = (e) => {
        if (tracing || drawing) return
        new maplibregl.Popup({ maxWidth: '280px', className: 'bw-popup' }).setLngLat(e.lngLat).setHTML(html(e.features[0].properties)).addTo(m)
      }
      const enter = () => { m.getCanvas().style.cursor = 'pointer' }
      const leave = () => { m.getCanvas().style.cursor = '' }
      m.on('click', lid, fn); m.on('mouseenter', lid, enter); m.on('mouseleave', lid, leave)
      return [lid, fn, enter, leave]
    })
    return () => fns.forEach(([lid, fn, enter, leave]) => { m.off('click', lid, fn); m.off('mouseenter', lid, enter); m.off('mouseleave', lid, leave) })
  }, [tracing, drawing])

  // flow-path trace (bonus)
  useEffect(() => {
    const m = map.current
    if (!m || !tracing || !runId) return
    const click = async (e) => {
      setTracing(false)
      try {
        const r = await postJSON(`${API}/runs/${runId}/trace`, { lon: e.lngLat.lng, lat: e.lngLat.lat })
        setTrace(r)
        m.getSource('trace').setData({ type: 'FeatureCollection', features: [r.path, ...r.settlements.map((s) => ({
          type: 'Feature', geometry: { type: 'Point', coordinates: [s.lon, s.lat] }, properties: s }))] })
        setTab('trace'); setDrawerOpen(true)
      } catch (err) { setError(err.message) }
    }
    m.once('click', click)
    return () => { m.off('click', click) }
  }, [tracing, runId])

  const flyTo = (lon, lat) => map.current?.flyTo({ center: [lon, lat], zoom: 13.5, duration: 900 })
  const showDrawer = Boolean(summary)
  const title = runTitle(summary?.bbox || bbox, caseStudy)

  return (
    <div className={`app ${showDrawer && drawerOpen ? 'has-drawer' : ''}`}>
      <main className={`map ${drawing || tracing ? 'picking' : ''}`}><div ref={mapEl} className="map-canvas" /></main>
      {(drawing || tracing) && <div className="hint glass">{drawing ? 'Click and drag on the map to draw the area' : 'Click a point upstream to trace the flood path'}<button className="link" onClick={() => { setDrawing(false); setTracing(false) }}>Cancel</button></div>}

      <header className="topbar glass">
        <div className="brand">
          <Logo />
          <div><h1>Baadhi Watch</h1><span>Flood damage from space</span></div>
        </div>
        {summary && <div className="run-pill"><span className="dot live" />{runTitle(summary.bbox, caseStudy)}<span className="sep">·</span>{fmtDate(summary.event_date)}</div>}
        <div className="top-actions">
          <button className={`ghost ${panel === 'history' ? 'on' : ''}`} onClick={() => setPanel(panel === 'history' ? null : 'history')}>
            <Icon name="history" />History{runs.some((r) => ACTIVE.includes(r.status)) && <span className="badge">{runs.filter((r) => ACTIVE.includes(r.status)).length}</span>}
          </button>
          <button className={`ghost ${panel === 'system' ? 'on' : ''}`} onClick={() => setPanel(panel === 'system' ? null : 'system')} title="System status">
            <span className={`dot ${health?.sentinel_hub ? 'ok' : 'bad'}`} />System
          </button>
          {panel === 'system' && <SystemStatus health={health} onClose={() => setPanel(null)} />}
        </div>
      </header>

      <section className={`card analyse glass ${cardOpen ? '' : 'collapsed'}`}>
        <button className="card-head" onClick={() => setCardOpen(!cardOpen)} aria-expanded={cardOpen}>
          <span className="card-title"><Icon name="satellite" />{running ? 'Analysis in progress' : 'Analyse a flood'}</span>
          <Icon name={cardOpen ? 'up' : 'down'} />
        </button>
        {cardOpen && (running ? (
          <Progress state={state} queuePos={runs.find((r) => r.id === runId)?.queue_position} onCancel={() => cancelRun(runId)} />
        ) : (
          <AnalyseForm bbox={bbox} setBbox={setBbox} date={date} setDate={setDate} drawing={drawing} setDrawing={setDrawing}
            onCaseStudy={() => { if (caseStudy) { setBbox(caseStudy.bbox); setDate(caseStudy.event_date) } }}
            onRun={() => start()} error={error} />
        ))}
      </section>

      {!summary && !running && !runId && <Welcome onCaseStudy={openCaseStudy} onDraw={() => { setDrawing(true); setCardOpen(true) }} />}

      {showDrawer && (drawerOpen ? (
        <aside className="drawer glass">
          <div className="drawer-head">
            <nav className="tabs">
              {TABS.map(([k, l]) => <button key={k} className={tab === k ? 'active' : ''} onClick={() => setTab(k)}>{l}</button>)}
            </nav>
            <button className="icon-btn" onClick={() => setDrawerOpen(false)} title="Hide results"><Icon name="right" /></button>
          </div>
          <div className="drawer-body">
            {tab === 'overview' && <Overview s={summary} runId={runId} flyTo={flyTo} />}
            {tab === 'report' && <Report runId={runId} llm={health?.llm} />}
            {tab === 'ask' && <Ask runId={runId} />}
            {tab === 'trace' && <Trace trace={trace} tracing={tracing} setTracing={setTracing} flyTo={flyTo} />}
            {tab === 'accuracy' && <Accuracy health={health} runId={runId} />}
          </div>
        </aside>
      ) : (
        <button className="drawer-tab glass" onClick={() => setDrawerOpen(true)}><Icon name="left" />Results</button>
      ))}

      {summary && (
        <div className="map-tools">
          <button className={`fab glass ${panel === 'layers' ? 'on' : ''}`} onClick={() => setPanel(panel === 'layers' ? null : 'layers')}><Icon name="layers" />Layers</button>
          <Legend />
          {panel === 'layers' && <LayersPanel visible={visible} setVisible={setVisible} opacity={opacity} setOpacity={setOpacity} onClose={() => setPanel(null)} />}
        </div>
      )}

      {panel === 'history' && <History runs={runs} caseStudy={caseStudy} currentId={runId} onOpen={openRun} onCancel={cancelRun} onDelete={deleteRun} onClose={() => setPanel(null)} />}

      <footer className="attrib">Contains modified Copernicus Sentinel data 2026 · Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the EU and ESA · © OpenStreetMap contributors · Educational prototype, not an operational tool</footer>
      <span className="sr-only">{title}</span>
    </div>
  )
}

// ------------------------------------------------------------------ left card
function AnalyseForm({ bbox, setBbox, date, setDate, drawing, setDrawing, onCaseStudy, onRun, error }) {
  const [edit, setEdit] = useState(false)
  const [kx, ky] = areaKm(bbox)
  const tooBig = bbox[2] - bbox[0] > 1 || bbox[3] - bbox[1] > 1
  return (
    <div className="card-body">
      <div className="field">
        <div className="field-label">Area</div>
        <div className="area-summary">
          <b>{fmt(kx, 0)} × {fmt(ky, 0)} km</b>
          <span>{bbox[1].toFixed(2)}–{bbox[3].toFixed(2)}°N · {bbox[0].toFixed(2)}–{bbox[2].toFixed(2)}°E</span>
        </div>
        <div className="btn-row">
          <button className={drawing ? 'on' : ''} onClick={() => setDrawing(!drawing)}><Icon name="draw" />{drawing ? 'Drawing…' : 'Draw on map'}</button>
          <button onClick={onCaseStudy}><Icon name="pin" />Trishuli case study</button>
        </div>
        <button className="link small" onClick={() => setEdit(!edit)}>{edit ? 'Hide coordinates' : 'Edit coordinates'}</button>
        {edit && (
          <div className="coords">
            {['West', 'South', 'East', 'North'].map((k, i) => (
              <label key={k}>{k}<input type="number" step="0.01" value={bbox[i]} onChange={(e) => { const b = [...bbox]; b[i] = +e.target.value; setBbox(b) }} /></label>
            ))}
          </div>
        )}
      </div>
      <div className="field">
        <label className="field-label" htmlFor="flood-date">Flood date</label>
        <input id="flood-date" type="date" value={date} onChange={(e) => setDate(e.target.value)} />
      </div>
      {tooBig && <p className="note warn">Areas larger than 1° (about 100 km) are not supported.</p>}
      <button className="primary" disabled={tooBig} onClick={onRun}>Run analysis</button>
      {error && <p className="note bad">{error}</p>}
      <p className="note">Uses Sentinel-1 radar, Sentinel-2 optical, the Copernicus DEM and pre-flood OpenStreetMap. New areas take a few minutes; finished runs open instantly.</p>
    </div>
  )
}

function Progress({ state, queuePos, onCancel }) {
  const [, tick] = useState(0)
  const [showLog, setShowLog] = useState(false)
  useEffect(() => { const t = setInterval(() => tick((x) => x + 1), 1000); return () => clearInterval(t) }, [])
  const idx = STEPS.findIndex(([k]) => k === state.step)
  const elapsed = state.started ? Math.max(0, Math.round((Date.now() - new Date(state.started)) / 1000)) : 0
  return (
    <div className="card-body">
      {state.status === 'queued' ? (
        <p className="note">Waiting in the queue{queuePos ? ` (position ${queuePos})` : ''}. Runs execute one at a time to protect the free satellite-processing quota.</p>
      ) : (
        <>
          <div className="progress"><div style={{ width: `${Math.max(4, state.progress || 0)}%` }} /></div>
          <div className="progress-meta"><span>{state.progress || 0}%</span><span>{Math.floor(elapsed / 60)}:{String(elapsed % 60).padStart(2, '0')} elapsed</span></div>
          <ol className="steps">
            {STEPS.map(([k, l], i) => (
              <li key={k} className={i < idx ? 'done' : i === idx ? 'now' : ''}><span className="step-dot" />{l}</li>
            ))}
          </ol>
        </>
      )}
      <div className="btn-row">
        <button onClick={onCancel}><Icon name="x" />Cancel run</button>
        <button onClick={() => setShowLog(!showLog)}>{showLog ? 'Hide details' : 'Show details'}</button>
      </div>
      {showLog && <pre className="log">{(state.log || []).slice(-10).join('\n')}</pre>}
    </div>
  )
}

function Welcome({ onCaseStudy, onDraw }) {
  return (
    <section className="welcome glass">
      <div className="eyebrow">Track B · Mapping flood damage from space</div>
      <h2>See where a flood hit and who is cut off, from free satellite data.</h2>
      <ul className="features">
        <li><span className="fi" style={{ '--c': '#4f8cff' }}><Icon name="satellite" /></span><div><b>Where did it hit?</b><span>Radar sees through monsoon cloud; AI and optical change confirm it.</span></div></li>
        <li><span className="fi" style={{ '--c': '#f59e0b' }}><Icon name="home" /></span><div><b>What was damaged?</b><span>Buildings, roads and bridges from the pre-flood map.</span></div></li>
        <li><span className="fi" style={{ '--c': '#ef4444' }}><Icon name="route" /></span><div><b>Who is cut off?</b><span>Villages that lost their road to a town or hospital, ranked by people.</span></div></li>
      </ul>
      <div className="btn-row">
        <button className="primary" onClick={onCaseStudy}>Open Trishuli case study</button>
        <button onClick={onDraw}>Draw your own area</button>
      </div>
    </section>
  )
}

// ------------------------------------------------------------------ drawer tabs
function Overview({ s, runId, flyTo }) {
  const [all, setAll] = useState(false)
  const c = s.network.counts
  const d = s.detect
  const b = s.buildings
  const km = Object.values(s.roads.km_by_class).reduce((a, v) => a + v, 0)
  const pri = s.network.settlements.filter((r) => r.status === 'cut_off' || r.status === 'detour')
  const file = (n) => `${API}/runs/${runId}/files/${n}`
  return (
    <div className="pad">
      <div className="hero">
        <div className="hero-num">{fmt(c.cut_off)}</div>
        <div className="hero-txt"><b>villages cut off</b><span>≈ {fmt(c.people_cut_off)} people without a road to a town or hospital</span></div>
      </div>
      <div className="kpis">
        <Kpi icon="home" v={fmt(b.in_water + b.in_debris)} l="buildings hit" sub={`+${fmt(b.at_risk)} within 30 m`} />
        <Kpi icon="water" v={fmt(d.affected_ha, 0)} l="hectares flooded" sub={`${fmt(d.high_conf_ha, 0)} ha high confidence`} />
        <Kpi icon="route" v={fmt(km, 0)} l="km of road blocked" sub={`${fmt(s.bridges.likely_damaged)} of ${fmt(s.bridges.total)} bridges hit`} />
        <Kpi icon="hospital" v={fmt(c.hospital_access_lost)} l="lost hospital access" sub={`${fmt(c.detour)} on long detours`} />
      </div>

      <div className="section-head"><h3>Priority villages</h3><span>{pri.length} need help · tap to zoom</span></div>
      {pri.length === 0 ? <p className="note">No village lost its road connection.</p> : (
        <ul className="villages">
          {(all ? pri : pri.slice(0, 8)).map((r, i) => (
            <li key={r.osm_id || r.name + i}>
              <button onClick={() => flyTo(r.lon, r.lat)}>
                <span className="rank" style={{ '--c': STATUS_COLOR[r.status] }}>{i + 1}</span>
                <span className="v-main"><b>{r.name_en || r.name}</b>
                  <span>{r.cause ? `Road breaks ${r.cause.km_along_route} km away${r.cause.bridge ? ' · bridge' : ''}` : STATUS_LABEL[r.status]}{r.name_en && r.name !== r.name_en ? ` · ${r.name}` : ''}</span></span>
                <span className="v-people"><b>{fmt(r.est_people)}</b><span>people</span></span>
              </button>
            </li>
          ))}
        </ul>
      )}
      {pri.length > 8 && <button className="link" onClick={() => setAll(!all)}>{all ? 'Show fewer' : `Show all ${pri.length}`}</button>}

      <div className="section-head"><h3>For field teams</h3></div>
      <div className="exports">
        <a href={file('priority.csv')} download><Icon name="download" />Priority list<small>CSV</small></a>
        <a href={file('cutoff.kml')} download><Icon name="download" />Cut-off villages<small>KML · Google Earth</small></a>
        <a href={file('flood.geojson')} download><Icon name="download" />Flood extent<small>GeoJSON</small></a>
        <a href={file('roads_blocked.geojson')} download><Icon name="download" />Blocked roads<small>GeoJSON</small></a>
      </div>

      <details className="inputs">
        <summary>Satellite data used</summary>
        <ul>
          {s.pairs.map((p) => <li key={p.post.label}><b>Radar after:</b> {p.post.label}<br /><b>before:</b> {p.pres.map((x) => x.label).join('; ')}</li>)}
          <li><b>Optical:</b> {s.s2_note}</li>
          <li><b>Map:</b> OpenStreetMap {s.osm_snapshot} ({s.osm_source})</li>
          <li><b>Terrain:</b> {s.detect.zone?.regime}, relief {fmt(s.detect.zone?.relief_m)} m</li>
          <li><b>Water model:</b> {s.model}</li>
        </ul>
      </details>
    </div>
  )
}

function Kpi({ icon, v, l, sub }) {
  return <div className="kpi"><span className="kpi-icon"><Icon name={icon} /></span><b>{v}</b><span>{l}</span><small>{sub}</small></div>
}

function Report({ runId, llm }) {
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
    w.document.write(`<html><head><meta charset="utf-8"><title>Situation report</title><style>body{font:13px/1.5 Inter,system-ui,sans-serif;max-width:760px;margin:28px auto;color:#111}table{border-collapse:collapse;width:100%;font-size:12px}td,th{border:1px solid #ddd;padding:4px 6px;text-align:left}h1{font-size:20px}h2{font-size:15px;margin-top:16px}</style></head><body>${marked.parse(doc.markdown)}</body></html>`)
    w.document.close(); w.focus(); w.print()
  }
  return (
    <div className="pad">
      <div className="toolbar">
        <Seg value={lang} onChange={setLang} options={[['en', 'English'], ['ne', 'नेपाली']]} />
        <Seg value={ai ? 'ai' : 'tpl'} onChange={(v) => setAi(v === 'ai')} options={[['tpl', 'Template'], ['ai', 'AI-written']]} disabled={!llm ? ['ai'] : []} />
      </div>
      {busy && <div className="skeleton"><span /><span /><span /><p className="note">{ai ? 'Writing the report. Every number is filled in from the map data…' : 'Loading…'}</p></div>}
      {doc && <>
        <article className="md paper" dangerouslySetInnerHTML={{ __html: marked.parse(doc.markdown) }} />
        <div className="report-foot"><span className="note">Source: {doc.source}</span><button onClick={print}><Icon name="print" />Print / PDF</button></div>
      </>}
      {!llm && <p className="note">AI-written reports need OPENROUTER_API_KEY or ANTHROPIC_API_KEY in .env.</p>}
    </div>
  )
}

function Ask({ runId }) {
  const [q, setQ] = useState('')
  const [lang, setLang] = useState('en')
  const [items, setItems] = useState([])
  const [busy, setBusy] = useState(false)
  const send = async (e, text) => {
    e?.preventDefault()
    const question = (text ?? q).trim()
    if (!question) return
    setQ(''); setBusy(true)
    try {
      const r = await postJSON(`${API}/runs/${runId}/ask`, { question, lang })
      setItems((it) => [...it, { q: question, ...r }])
    } catch (err) { setItems((it) => [...it, { q: question, markdown: err.message, source: 'error' }]) }
    setBusy(false)
  }
  const examples = lang === 'ne' ? ['कति बस्ती सम्पर्कविहीन छन्?', 'कति पुल क्षतिग्रस्त छन्?'] :
    ['Which villages should helicopters reach first?', 'How many bridges are damaged?', 'How many people live in hit buildings?']
  return (
    <div className="pad chat">
      <div className="toolbar"><Seg value={lang} onChange={setLang} options={[['en', 'English'], ['ne', 'नेपाली']]} /></div>
      {items.length === 0 && <p className="note">Answers come only from this run's numbers. The assistant writes placeholders and the system fills in each value, so it cannot invent figures.</p>}
      <div className="bubbles">
        {items.map((it, i) => (
          <div key={i} className="qa">
            <div className="bubble me">{it.q}</div>
            <div className="bubble bot"><div className="md" dangerouslySetInnerHTML={{ __html: marked.parse(it.markdown) }} /><small>{it.source}</small></div>
          </div>
        ))}
        {busy && <div className="bubble bot typing"><span /><span /><span /></div>}
      </div>
      {items.length === 0 && <div className="examples">{examples.map((x) => <button key={x} onClick={() => send(null, x)}>{x}</button>)}</div>}
      <form onSubmit={send} className="ask"><input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask about this flood…" /><button className="primary" disabled={busy}><Icon name="send" /></button></form>
    </div>
  )
}

function Trace({ trace, tracing, setTracing, flyTo }) {
  const RISK = { high: '#ef4444', medium: '#f59e0b', low: '#22c55e' }
  return (
    <div className="pad">
      <p className="note">Pick any point upstream. The system follows the downhill flow path on the Copernicus DEM and lists villages along it in the order the water would reach them.</p>
      <button className="primary" onClick={() => setTracing(!tracing)}><Icon name="pin" />{tracing ? 'Click a point on the map…' : 'Pick upstream point'}</button>
      {trace && <>
        <div className="kpis three">
          <div className="kpi"><b>{trace.path.properties.length_km}</b><span>km traced</span></div>
          <div className="kpi"><b>{fmt(trace.path.properties.drop_m)}</b><span>m drop</span></div>
          <div className="kpi"><b>{fmt(trace.buildings_near_river)}</b><span>houses near river</span></div>
        </div>
        <ul className="villages">
          {trace.settlements.map((s) => (
            <li key={s.name + s.km_downstream}>
              <button onClick={() => flyTo(s.lon, s.lat)}>
                <span className="rank km" style={{ '--c': RISK[s.risk] || '#94a3b8' }}>{s.km_downstream}<small>km</small></span>
                <span className="v-main"><b>{s.name_en || s.name}</b><span>{s.height_above_river_m ?? '–'} m above river · {fmt(s.buildings_near_river)} houses near river</span></span>
                <span className="v-people"><b>{s.eta_min[0]}–{s.eta_min[1]}</b><span>min*</span></span>
              </button>
            </li>
          ))}
        </ul>
        <p className="note">* {trace.note}</p>
      </>}
    </div>
  )
}

function Accuracy({ health, runId }) {
  const [emsr, setEmsr] = useState(null)
  useEffect(() => {
    getJSON(`${API}/runs/${runId}/files/emsr927_eval.json`).then(setEmsr).catch(() => setEmsr(null))
  }, [runId])
  const m = health?.model_metrics
  const e = emsr?.summary
  return (
    <div className="pad">
      <div className="section-head"><h3>Against the official Copernicus maps</h3><span>EMSR927 · checking only</span></div>
      {e ? (
        <>
          <div className="kpis">
            <Kpi icon="target" v={e.extent.f1} l="F1 score (flood extent)" sub={`IoU ${e.extent.iou}`} />
            <Kpi icon="home" v={`${Math.round(e.buildings?.recall * 100)}%`} l="EMS-damaged buildings found" sub={`${fmt(e.buildings?.inside_our_affected_area)} of ${fmt(e.buildings?.ems_damaged_or_destroyed)}`} />
          </div>
          <div className="bars">
            <Bar label="Precision" v={e.extent.precision} hint="share of our flood area that EMS also mapped" />
            <Bar label="Recall" v={e.extent.recall} hint="share of EMS flood area we found" />
            <Bar label="Recall (±2 px)" v={e.extent_2px_tolerance.recall} hint="allowing for 20 m pixels vs sub-metre imagery" />
          </div>
          {e.roads && <p className="note">Blocked roads: {e.roads.our_blocked_km_matching_ems} of our {e.roads.our_blocked_km} km match EMS-damaged roads (EMS total {e.roads.ems_affected_km} km). Turn on the <b>Agreement with EMSR927</b> layer to see where we agree.</p>}
        </>
      ) : <p className="note">No EMSR927 comparison for this area. Run <code>python -m eval.evaluate_emsr --run {runId}</code> after placing the EMS vectors in <code>validation/</code>.</p>}

      <div className="section-head"><h3>AI water model</h3><span>U-Net · Sen1Floods11</span></div>
      {m ? (
        <div className="bars">
          <Bar label="Test chips" v={m.test?.iou_water} hint={`IoU · F1 ${m.test?.f1_water}`} />
          <Bar label="Bolivia (unseen event)" v={m.bolivia?.iou_water} hint={`IoU · F1 ${m.bolivia?.f1_water}`} />
        </div>
      ) : <p className="note">No model metrics found.</p>}
      <p className="note">The model was trained on 446 hand-labelled Sentinel-1 flood images from 11 countries and tested on scenes it never saw.</p>
    </div>
  )
}

function Bar({ label, v, hint }) {
  return (
    <div className="bar">
      <div className="bar-top"><span>{label}</span><b>{v != null ? v.toFixed(2) : '–'}</b></div>
      <div className="bar-track"><div style={{ width: `${(v || 0) * 100}%` }} /></div>
      {hint && <small>{hint}</small>}
    </div>
  )
}

// ------------------------------------------------------------------ popovers
function LayersPanel({ visible, setVisible, opacity, setOpacity, onClose }) {
  return (
    <div className="popover layers glass">
      <div className="pop-head"><b>Map layers</b><button className="icon-btn" onClick={onClose}><Icon name="x" /></button></div>
      {LAYER_GROUPS.map(([g, ids]) => (
        <div key={g} className="group">
          <div className="group-label">{g}</div>
          {ids.map((id) => (
            <label key={id} className="switch">
              <input type="checkbox" checked={visible[id]} onChange={() => setVisible({ ...visible, [id]: !visible[id] })} />
              <span className="track" /><span>{LABEL[id]}</span>
            </label>
          ))}
        </div>
      ))}
      <label className="range">Imagery opacity<input type="range" min="0" max="1" step="0.05" value={opacity} onChange={(e) => setOpacity(+e.target.value)} /></label>
    </div>
  )
}

function History({ runs, caseStudy, currentId, onOpen, onCancel, onDelete, onClose }) {
  return (
    <div className="modal-wrap" onClick={onClose}>
      <div className="modal glass" onClick={(e) => e.stopPropagation()}>
        <div className="pop-head"><b>Run history</b><button className="icon-btn" onClick={onClose}><Icon name="x" /></button></div>
        <p className="note">Runs execute one at a time, so new ones wait in a queue. A run interrupted by a server restart can simply be run again.</p>
        {runs.length === 0 && <p className="note">No runs yet.</p>}
        <ul className="runs">
          {runs.map((r) => {
            const active = ACTIVE.includes(r.status)
            const [kx, ky] = areaKm(r.bbox)
            return (
              <li key={r.id} className={r.id === currentId ? 'current' : ''}>
                <div className="run-main">
                  <b>{runTitle(r.bbox, caseStudy)}</b>
                  <span>{fmtDate(r.event_date)} · {fmt(kx, 0)} × {fmt(ky, 0)} km{r.seconds ? ` · ${Math.round(r.seconds / 60)} min` : ''}</span>
                  {r.status === 'queued' && r.queue_position && <span>Position {r.queue_position} in queue</span>}
                  {r.status === 'running' && <span>{STEPS.find(([k]) => k === r.step)?.[1] || 'Starting'} · {r.progress || 0}%</span>}
                  {['error', 'interrupted', 'cancelled'].includes(r.status) && r.error && <span className="err">{r.error.slice(0, 140)}</span>}
                </div>
                <span className={`status ${r.status}`}>{r.status}</span>
                <div className="run-actions">
                  <button onClick={() => onOpen(r)}>{r.status === 'done' ? 'Open' : active ? 'View' : 'Run again'}</button>
                  {active ? <button onClick={() => onCancel(r.id)}>Cancel</button> : <button className="danger" onClick={() => onDelete(r.id)} title="Delete run"><Icon name="trash" /></button>}
                </div>
              </li>
            )
          })}
        </ul>
      </div>
    </div>
  )
}

function SystemStatus({ health }) {
  const rows = [
    ['Sentinel Hub (satellite data)', health?.sentinel_hub],
    [health?.ohsome ? 'Map data: ohsome API (worldwide)' : 'Map data: Nepal stored · elsewhere public servers', health?.ohsome],
    [`AI copilot${health?.llm_provider ? ` (${health.llm_provider})` : ''}`, health?.llm],
    [health?.model || 'Water model', health?.model?.startsWith('U-Net')],
  ]
  return (
    <div className="popover system glass">
      {rows.map(([l, ok]) => <div key={l} className="sys-row"><span className={`dot ${ok ? 'ok' : 'bad'}`} />{l}</div>)}
      {health?.model_metrics?.bolivia && <div className="sys-row muted">Model IoU on an unseen flood: {health.model_metrics.bolivia.iou_water}</div>}
    </div>
  )
}

function Legend() {
  const items = [['#3b82f6', 'Flood water', 'sq'], ['#c2702e', 'Debris', 'sq'], ['#ef4444', 'Blocked road', 'ln'],
    [STATUS_COLOR.cut_off, 'Cut off', 'ci'], [STATUS_COLOR.detour, 'Detour', 'ci'], [STATUS_COLOR.connected, 'Connected', 'ci']]
  return <div className="legend glass">{items.map(([c, l, t]) => <span key={l}><i className={t} style={{ background: c }} />{l}</span>)}</div>
}

function Seg({ value, onChange, options, disabled = [] }) {
  return (
    <div className="seg">
      {options.map(([k, l]) => <button key={k} className={value === k ? 'on' : ''} disabled={disabled.includes(k)} onClick={() => onChange(k)}>{l}</button>)}
    </div>
  )
}

function Logo() {
  return (
    <svg width="30" height="30" viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="9" fill="url(#lg)" />
      <defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stopColor="#2f6bff" /><stop offset="1" stopColor="#13b5c8" /></linearGradient></defs>
      <path d="M5 20c3-3 6 3 9 0s6 3 9 0 4-1.5 4-1.5V25H5z" fill="#fff" opacity=".95" />
      <circle cx="22" cy="10" r="3.2" fill="#ffd166" />
    </svg>
  )
}

const PATHS = {
  satellite: 'M7 7l3 3m4 4l3 3M5 9l4-4 3 3-4 4zm7 7l4-4 3 3-4 4zM12 12l2 2m4-9a4 4 0 0 1 4 4m-4-7a7 7 0 0 1 7 7',
  history: 'M3 12a9 9 0 1 0 3-6.7L3 8m0-5v5h5m4-1v5l3 2',
  layers: 'M12 3l9 5-9 5-9-5zm-9 9l9 5 9-5m-18 4l9 5 9-5',
  draw: 'M4 20h4L18.5 9.5a2.8 2.8 0 0 0-4-4L4 16zm9.5-13.5l4 4',
  pin: 'M12 21s-7-6-7-11a7 7 0 0 1 14 0c0 5-7 11-7 11zm0-8a3 3 0 1 0 0-6 3 3 0 0 0 0 6z',
  home: 'M3 11l9-7 9 7v9a1 1 0 0 1-1 1h-5v-6h-6v6H4a1 1 0 0 1-1-1z',
  route: 'M6 19a2 2 0 1 0 0-4 2 2 0 0 0 0 4zm12-10a2 2 0 1 0 0-4 2 2 0 0 0 0 4zM8 17h7a3 3 0 0 0 0-6H9a3 3 0 0 1 0-6h7',
  water: 'M12 3s6 7 6 11a6 6 0 0 1-12 0c0-4 6-11 6-11z',
  hospital: 'M4 4h16v16H4zm8 4v8m-4-4h8',
  target: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zm0-5a4 4 0 1 0 0-8 4 4 0 0 0 0 8zm0-3a1 1 0 1 0 0-2 1 1 0 0 0 0 2z',
  download: 'M12 4v11m-5-5l5 5 5-5M5 20h14',
  print: 'M7 9V3h10v6M7 18H5a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2M7 14h10v7H7z',
  send: 'M4 12l16-8-6 16-2-7z',
  trash: 'M4 7h16M10 11v6m4-6v6M6 7l1 13h10l1-13M9 7V4h6v3',
  x: 'M6 6l12 12M18 6L6 18', up: 'M6 15l6-6 6 6', down: 'M6 9l6 6 6-6', left: 'M15 6l-6 6 6 6', right: 'M9 6l6 6-6 6',
}
function Icon({ name }) {
  return <svg className="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={PATHS[name]} /></svg>
}

// ------------------------------------------------------------------ helpers
function emptyFC() { return { type: 'FeatureCollection', features: [] } }
function norm(a, b) { return [Math.min(a.lng, b.lng), Math.min(a.lat, b.lat), Math.max(a.lng, b.lng), Math.max(a.lat, b.lat)] }
function bboxFC([w, s, e, n]) {
  return { type: 'FeatureCollection', features: [{ type: 'Feature', properties: {}, geometry: { type: 'Polygon', coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] } }] }
}
function fitPadding(m, left, right) {
  // leave room for the floating panels on wide screens; on narrow screens the panels sit below the map
  const wide = m.getCanvas().clientWidth > 1000 && m.getCanvas().clientHeight > 500
  return wide ? { top: 90, bottom: 70, left, right } : 20
}
function areaKm([w, s, e, n]) {
  const lat = ((s + n) / 2) * Math.PI / 180
  return [(e - w) * 111.32 * Math.cos(lat), (n - s) * 110.57]
}
function runTitle(bbox, caseStudy) {
  if (caseStudy && bbox.every((v, i) => Math.abs(v - caseStudy.bbox[i]) < 1e-3)) return 'Trishuli case study'
  const [w, s, e, n] = bbox
  return `Area near ${((s + n) / 2).toFixed(2)}°N, ${((w + e) / 2).toFixed(2)}°E`
}
function fmtDate(d) {
  return new Date(`${d}T00:00:00Z`).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' })
}
function layerIds(id) {
  if (RASTERS.some(([r]) => r === id)) return [`r-${id}`]
  if (id === 'settlements') return ['v-settlements', 'v-settlements-label']
  if (id === 'targets') return ['v-targets', 'v-targets-label']
  if (id === 'flood') return ['v-flood', 'v-flood-line']
  return [`v-${id}`]
}
function clearRunLayers(m) {
  if (!m?.getStyle()) return
  for (const lid of m.getStyle().layers.map((l) => l.id)) if (/^(r-|v-|t-)/.test(lid)) m.removeLayer(lid)
  for (const sid of Object.keys(m.getStyle().sources)) if (sid.startsWith('src-')) m.removeSource(sid)
}
function addRunLayers(m, s, visible, opacity) {
  const base = `${API}/runs/${s.id}/files/`
  clearRunLayers(m)
  const vis = (id) => (visible[id] ? 'visible' : 'none')
  for (const [id] of [...RASTERS].reverse()) {
    m.addSource(`src-${id}`, { type: 'image', url: base + `${id}.png`, coordinates: s.grid.corners })
    m.addLayer({ id: `r-${id}`, type: 'raster', source: `src-${id}`, layout: { visibility: vis(id) }, paint: { 'raster-opacity': opacity, 'raster-fade-duration': 0 } })
  }
  const gj = (id) => m.addSource(`src-${id}`, { type: 'geojson', data: base + `${id}.geojson` })
  ;['roads', 'flood', 'buildings_affected', 'roads_blocked', 'bridges', 'breaks', 'targets', 'settlements'].forEach(gj)
  m.addLayer({ id: 'v-roads', type: 'line', source: 'src-roads', layout: { visibility: vis('roads') }, paint: { 'line-color': '#334155', 'line-width': ['interpolate', ['linear'], ['zoom'], 9, 0.5, 14, 2], 'line-opacity': 0.6 } })
  m.addLayer({ id: 'v-flood', type: 'fill', source: 'src-flood', layout: { visibility: vis('flood') }, paint: {
    'fill-color': ['match', ['get', 'class'], 'water', '#3b82f6', '#c2702e'],
    'fill-opacity': ['match', ['get', 'confidence'], 'high', 0.72, 'medium', 0.55, 0.35] } })
  m.addLayer({ id: 'v-flood-line', type: 'line', source: 'src-flood', layout: { visibility: vis('flood') }, paint: { 'line-color': ['match', ['get', 'class'], 'water', '#1d4ed8', '#8a4513'], 'line-width': 0.8 } })
  m.addLayer({ id: 'v-buildings_affected', type: 'fill', source: 'src-buildings_affected', layout: { visibility: vis('buildings_affected') }, paint: {
    'fill-color': ['match', ['get', 'status'], 'in_water', '#1e3a8a', 'in_debris', '#7c2d12', '#f97316'], 'fill-opacity': 0.9 } })
  m.addLayer({ id: 'v-roads_blocked', type: 'line', source: 'src-roads_blocked', layout: { visibility: vis('roads_blocked'), 'line-cap': 'round' }, paint: { 'line-color': '#ef4444', 'line-width': ['interpolate', ['linear'], ['zoom'], 9, 2.5, 14, 6] } })
  m.addLayer({ id: 'v-bridges', type: 'line', source: 'src-bridges', layout: { visibility: vis('bridges') }, paint: {
    'line-color': ['match', ['get', 'status'], 'likely_damaged', '#ef4444', '#0f172a'], 'line-width': 6, 'line-dasharray': [1, 0.6] } })
  m.addLayer({ id: 'v-breaks', type: 'circle', source: 'src-breaks', layout: { visibility: vis('breaks') }, paint: { 'circle-radius': 5, 'circle-color': '#fff', 'circle-stroke-color': '#ef4444', 'circle-stroke-width': 3 } })
  m.addLayer({ id: 'v-targets', type: 'circle', source: 'src-targets', layout: { visibility: vis('targets') }, paint: { 'circle-radius': 7, 'circle-color': ['match', ['get', 'kind'], 'hospital', '#e11d48', '#0f172a'], 'circle-stroke-color': '#fff', 'circle-stroke-width': 2 } })
  m.addLayer({ id: 'v-targets-label', type: 'symbol', source: 'src-targets', minzoom: 10, layout: { visibility: vis('targets'), 'text-field': ['concat', ['match', ['get', 'kind'], 'hospital', '✚ ', ''], ['get', 'name']], 'text-size': 11, 'text-offset': [0, 1.2], 'text-anchor': 'top', 'text-font': ['Noto Sans Regular'] }, paint: { 'text-color': '#0f172a', 'text-halo-color': '#fff', 'text-halo-width': 1.5 } })
  const affected = ['in', ['get', 'status'], ['literal', ['cut_off', 'detour']]]
  m.addLayer({ id: 'v-settlements', type: 'circle', source: 'src-settlements', layout: { visibility: vis('settlements'), 'circle-sort-key': ['case', affected, 1, 0] }, paint: {
    'circle-radius': ['case', affected, ['interpolate', ['linear'], ['get', 'est_people'], 0, 5, 500, 8, 3000, 13], 3.5],
    'circle-color': ['match', ['get', 'status'], 'cut_off', STATUS_COLOR.cut_off, 'detour', STATUS_COLOR.detour, 'connected', STATUS_COLOR.connected, STATUS_COLOR.no_road_pre],
    'circle-opacity': ['case', affected, 1, 0.55],
    'circle-stroke-color': '#fff', 'circle-stroke-width': ['case', affected, 2, 0.8], 'circle-stroke-opacity': ['case', affected, 1, 0.6] } })
  m.addLayer({ id: 'v-settlements-label', type: 'symbol', source: 'src-settlements', minzoom: 10.5, filter: ['in', ['get', 'status'], ['literal', ['cut_off', 'detour']]], layout: { visibility: vis('settlements'), 'text-field': ['coalesce', ['get', 'name_en'], ['get', 'name']], 'text-size': 11, 'text-offset': [0, 1], 'text-anchor': 'top', 'text-font': ['Noto Sans Regular'] }, paint: { 'text-color': '#0f172a', 'text-halo-color': '#fff', 'text-halo-width': 1.4 } })
  if (!m.getLayer('t-trace')) {
    m.addLayer({ id: 't-trace', type: 'line', source: 'trace', filter: ['==', '$type', 'LineString'], paint: { 'line-color': '#8b5cf6', 'line-width': 4 } })
    m.addLayer({ id: 't-trace-pts', type: 'circle', source: 'trace', filter: ['==', '$type', 'Point'], paint: { 'circle-radius': 6, 'circle-color': '#8b5cf6', 'circle-stroke-color': '#fff', 'circle-stroke-width': 2 } })
  }
}
