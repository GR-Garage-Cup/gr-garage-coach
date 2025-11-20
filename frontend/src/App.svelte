<script>
  import { onMount } from 'svelte';
  import Landing from './Landing.svelte';
  import TrackMap from './TrackMap.svelte';
  import SpeedTrace from './SpeedTrace.svelte';
  import TelemetryTraces from './TelemetryTraces.svelte';
  import TractionCircle from './TractionCircle.svelte';

  let showLanding = true;
  let view = 'upload';
  let data = null;
  let vizData = null;
  let file = null;
  let loading = false;
  let error = null;
  let online = false;
  let selectedTrack = 'road_america';
  let driverId = 'DRIVER_001';

  const API = 'https://gr-garage-coach-api-ce18088c2f04.herokuapp.com';

  const TRACKS = [
    { key: 'road_america', name: 'Road America' },
    { key: 'cota', name: 'Circuit of the Americas' },
    { key: 'barber', name: 'Barber Motorsports Park' },
    { key: 'indianapolis', name: 'Indianapolis Road Course' },
    { key: 'sebring', name: 'Sebring International Raceway' },
    { key: 'sonoma', name: 'Sonoma Raceway' },
    { key: 'vir', name: 'Virginia International Raceway' }
  ];

  onMount(async () => {
    try {
      const res = await fetch(`${API}/health`);
      online = res.ok;
    } catch (e) {
      online = false;
    }
  });

  function handleFile(e) {
    file = e.target.files[0];
    error = null;
  }

  async function analyze() {
    if (!file) return;
    loading = true;
    error = null;

    try {
      const form = new FormData();
      form.append('file', file);
      form.append('track_name', selectedTrack);
      form.append('driver_id', driverId);

      const res = await fetch(`${API}/api/v1/analyze`, {
        method: 'POST',
        body: form
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Analysis failed');
      }

      data = await res.json();

      if (data.visualization_data) {
        vizData = data.visualization_data;
      }

      view = 'results';
    } catch (e) {
      error = e.message;
    } finally {
      loading = false;
    }
  }

  function reset() {
    view = 'upload';
    data = null;
    vizData = null;
    file = null;
  }

  function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(3);
    return `${mins}:${secs.padStart(6, '0')}`;
  }

  function formatDelta(seconds) {
    const sign = seconds >= 0 ? '+' : '';
    return `${sign}${seconds.toFixed(3)}s`;
  }
</script>

{#if showLanding}
  <Landing onEnter={() => showLanding = false} />
{:else if view === 'upload'}
<div class="page">
  <div class="content">
    <div class="header">
      <h1>GR GARAGE COACH</h1>
      <p class="subtitle">Upload your lap and let's see where you can get faster</p>
    </div>

    <div class="upload-section">
      <div class="friendly-intro">
        <p>
          Pick the track you ran, drop your TRD CSV file, and we'll break down your lap
          corner by corner.
        </p>
      </div>

      <div class="track-picker">
        <label for="track-select">Which track?</label>
        <select
          id="track-select"
          bind:value={selectedTrack}
          disabled={!online}
        >
          {#each TRACKS as track}
            <option value={track.key}>{track.name}</option>
          {/each}
        </select>
      </div>

      <div class="file-drop">
        <input
          type="file"
          accept=".csv"
          on:change={handleFile}
          id="file-input"
          disabled={!online}
        />
        <label for="file-input" class="drop-zone {!online ? 'disabled' : ''}">
          {#if file}
            <div class="file-selected">
              <div class="check">✓</div>
              <div>
                <div class="file-name">{file.name}</div>
                <div class="file-size">{(file.size / 1024).toFixed(1)} KB</div>
              </div>
            </div>
          {:else}
            <div class="drop-prompt">
              <div class="drop-icon">↑</div>
              <div class="drop-text">Click to select your TRD CSV file</div>
              <div class="drop-hint">Should have GPS, speed, brake, throttle data</div>
            </div>
          {/if}
        </label>
      </div>

      {#if error}
        <div class="error-box">
          <p><strong>Something went wrong:</strong></p>
          <p>{error}</p>
        </div>
      {/if}

      <button
        class="analyze-button"
        on:click={analyze}
        disabled={!file || loading || !online}
      >
        {loading ? 'Running the numbers...' : 'Analyze this lap'}
      </button>

      {#if !online}
        <div class="offline-note">
          <p>Backend is offline. Start the API server to analyze laps.</p>
        </div>
      {/if}
    </div>
  </div>
</div>

{:else}
<div class="page results-page">
  <div class="content">
    <div class="header">
      <h1>Here's what we found</h1>
      <button class="new-lap-btn" on:click={reset}>← Analyze another lap</button>
    </div>

    <div class="lap-summary">
      <div class="summary-intro">
        <p>
          You ran a <strong>{formatTime(data.lap_time)}</strong> at {TRACKS.find(t => t.key === selectedTrack)?.name}.
          Based on GPS physics and champion baseline data, you've got room for about <strong class="time-gain">{formatDelta(data.time_delta)}</strong>
          of improvement. Here's where it's hiding.
        </p>
      </div>

      <div class="time-cards">
        <div class="time-card">
          <label>Your lap time</label>
          <div class="time-big">{formatTime(data.lap_time)}</div>
        </div>
        <div class="time-card optimal">
          <label>Physics-based optimal</label>
          <div class="time-big">{formatTime(data.optimal_lap_time)}</div>
          <div class="time-note">Using your actual GPS and GR86 limits</div>
        </div>
        <div class="time-card delta">
          <label>What you're leaving out there</label>
          <div class="time-big">{formatDelta(data.time_delta)}</div>
          <div class="time-note">Real improvement potential, not guesswork</div>
        </div>
      </div>
    </div>

    <div class="driver-profile">
      <h2>How you drive</h2>
      <p class="section-intro">
        We analyzed 50+ measurements from your telemetry and figured out your driving style.
        Different styles need different coaching, so this matters.
      </p>

      <div class="profile-box">
        <div class="archetype-badge">{data.driver_archetype}</div>
        <div class="confidence-note">
          We're {(data.driver_confidence * 100).toFixed(0)}% confident on this classification
        </div>

        <div class="traits-grid">
          <div class="traits-col">
            <label>What you're good at:</label>
            {#each data.driver_strengths as strength}
              <div class="trait strength">{strength}</div>
            {/each}
          </div>
          <div class="traits-col">
            <label>What needs work:</label>
            {#each data.driver_weaknesses as weakness}
              <div class="trait weakness">{weakness}</div>
            {/each}
          </div>
        </div>
      </div>
    </div>

    <div class="performance-section">
      <h2>Your performance breakdown</h2>
      <p class="section-intro">
        These scores compare you to championship baseline data (top 10% of laps from actual GR Cup races).
      </p>

      <div class="metrics-list">
        <div class="metric-row">
          <div class="metric-info">
            <label>Braking</label>
            <p>How hard and late you're hitting the brakes</p>
          </div>
          <div class="metric-bar">
            <div class="bar-fill" style="width: {Math.min(data.overall_metrics.braking * 100, 100)}%"></div>
          </div>
          <div class="metric-score">{(data.overall_metrics.braking * 100).toFixed(0)}%</div>
        </div>

        <div class="metric-row">
          <div class="metric-info">
            <label>Acceleration</label>
            <p>Throttle application and traction out of corners</p>
          </div>
          <div class="metric-bar">
            <div class="bar-fill" style="width: {Math.min(data.overall_metrics.acceleration * 100, 100)}%"></div>
          </div>
          <div class="metric-score">{(data.overall_metrics.acceleration * 100).toFixed(0)}%</div>
        </div>

        <div class="metric-row">
          <div class="metric-info">
            <label>Cornering</label>
            <p>Apex speed and traction circle usage</p>
          </div>
          <div class="metric-bar">
            <div class="bar-fill" style="width: {Math.min(data.overall_metrics.cornering * 100, 100)}%"></div>
          </div>
          <div class="metric-score">{(data.overall_metrics.cornering * 100).toFixed(0)}%</div>
        </div>

        <div class="metric-row">
          <div class="metric-info">
            <label>Consistency</label>
            <p>How repeatable your inputs and lines are</p>
          </div>
          <div class="metric-bar">
            <div class="bar-fill" style="width: {Math.min(data.overall_metrics.consistency * 100, 100)}%"></div>
          </div>
          <div class="metric-score">{(data.overall_metrics.consistency * 100).toFixed(0)}%</div>
        </div>
      </div>
    </div>

    {#if vizData}
    <div class="visualizations-section">
      <h2>Telemetry data</h2>
      <p class="section-intro">
        Your actual GPS racing line, speed traces, and traction data from the lap you just uploaded.
        This is real telemetry, not simulations.
      </p>

      <div class="viz-grid">
        <TrackMap
          x={vizData.track_map.x}
          y={vizData.track_map.y}
          title="TRACK MAP - GPS RACING LINE"
        />

        <TractionCircle
          longitudinalG={vizData.traction_circle.longitudinal_g}
          lateralG={vizData.traction_circle.lateral_g}
          title="TRACTION CIRCLE (G-G DIAGRAM)"
        />
      </div>

      <SpeedTrace
        arcLength={vizData.speed_trace.arc_length}
        speed={vizData.speed_trace.speed}
        title="SPEED TRACE BY DISTANCE"
      />

      <TelemetryTraces
        arcLength={vizData.input_traces.arc_length}
        throttle={vizData.input_traces.throttle}
        brakeFront={vizData.input_traces.brake_front}
        brakeRear={vizData.input_traces.brake_rear}
        title="BRAKE & THROTTLE INPUTS"
      />
    </div>
    {/if}

    <div class="improvements-section">
      <h2>Where the time is</h2>
      <p class="section-intro">
        These are the biggest opportunities, sorted by potential time gain. Focus on the top few.
      </p>

      <div class="improvements-list">
        {#each data.top_improvements.slice(0, 5) as improvement, i}
          <div class="improvement-card">
            <div class="improvement-header">
              <span class="priority-badge">#{i + 1}</span>
              <span class="gain">{formatDelta(improvement.time_gain)}</span>
            </div>
            <div class="improvement-text">{improvement.description}</div>
          </div>
        {/each}
      </div>
    </div>

    <div class="training-section">
      <h2>What to practice</h2>
      <p class="section-intro">
        Here's your training plan. These drills are generated from your actual telemetry,
        not some generic template. GPS positions you can find on track, speed targets you can hit,
        traction numbers you can measure.
      </p>

      <div class="drills-list">
        {#each data.training_curriculum as drill, i}
          <div class="drill-card">
            <div class="drill-number">Drill {i + 1}</div>
            <div class="drill-content">{drill}</div>
          </div>
        {/each}
      </div>
    </div>

    <div class="skill-gaps-section">
      <h2>Skill gaps identified</h2>
      <p class="section-intro">
        Based on comparing your lap to champion baselines, here's what's holding you back.
      </p>

      <div class="gaps-list">
        {#each data.skill_gaps as gap}
          <div class="gap-item">{gap}</div>
        {/each}
      </div>
    </div>

    <div class="next-steps">
      <h2>Next steps</h2>
      <div class="steps-box">
        <p>
          Pick one or two drills from the training plan and run them at your next track day.
          Don't try to fix everything at once. Work on the top improvement first, get that
          dialed in, then come back and upload another lap to see what's next.
        </p>
        <p>
          That's kaizen. Continuous improvement through doing, not theory.
        </p>
      </div>
      <button class="new-lap-btn big" on:click={reset}>Analyze another lap</button>
    </div>
  </div>
</div>
{/if}

<style>
  :global(*) {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  .page {
    background: #000000;
    color: #e0e0e0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'SF Pro Text', 'Helvetica Neue', sans-serif;
    min-height: 100vh;
    padding: 3rem 2rem;
    position: relative;
  }

  .page::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: repeating-linear-gradient(
      0deg,
      rgba(255,255,255,0.01) 0px,
      rgba(255,255,255,0.01) 1px,
      transparent 1px,
      transparent 2px
    );
    pointer-events: none;
    z-index: 1;
  }

  .content {
    max-width: 900px;
    margin: 0 auto;
    position: relative;
    z-index: 2;
  }

  .header {
    margin-bottom: 3rem;
    padding-bottom: 1.5rem;
    border-bottom: 2px solid #ff0000;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  h1 {
    font-size: 2rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 0.02em;
  }

  h1::first-letter {
    color: #ff0000;
  }

  .subtitle {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.6);
    margin-top: 0.5rem;
  }

  h2 {
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 1rem;
    position: relative;
    padding-left: 1rem;
  }

  h2::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 3px;
    background: #ff0000;
  }

  .section-intro {
    font-size: 1.05rem;
    line-height: 1.7;
    color: rgba(255,255,255,0.8);
    margin-bottom: 2rem;
  }

  /* UPLOAD VIEW */
  .friendly-intro {
    background: rgba(255,255,255,0.02);
    border-left: 3px solid rgba(255,255,255,0.1);
    padding: 1.5rem;
    margin-bottom: 2rem;
  }

  .friendly-intro p {
    font-size: 1.05rem;
    line-height: 1.7;
    color: rgba(255,255,255,0.85);
  }

  .track-picker {
    margin-bottom: 2rem;
  }

  .track-picker label {
    display: block;
    font-size: 1.1rem;
    font-weight: 600;
    color: rgba(255,255,255,0.9);
    margin-bottom: 0.75rem;
  }

  .track-picker select {
    width: 100%;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.15);
    color: #ffffff;
    padding: 1rem;
    font-size: 1rem;
    outline: none;
    cursor: pointer;
    font-family: inherit;
  }

  .track-picker select:hover:not(:disabled) {
    border-color: rgba(255,0,0,0.5);
  }

  .track-picker select:focus {
    border-color: #ff0000;
    box-shadow: 0 0 0 2px rgba(255,0,0,0.2);
  }

  .track-picker select:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .file-drop {
    margin-bottom: 2rem;
  }

  #file-input {
    display: none;
  }

  .drop-zone {
    display: block;
    border: 2px dashed rgba(255,255,255,0.2);
    padding: 3rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    background: rgba(255,255,255,0.02);
  }

  .drop-zone:hover:not(.disabled) {
    border-color: #ff0000;
    background: rgba(255,0,0,0.05);
  }

  .drop-zone.disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .drop-prompt {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.75rem;
  }

  .drop-icon {
    font-size: 3rem;
    color: rgba(255,255,255,0.3);
  }

  .drop-text {
    font-size: 1.1rem;
    font-weight: 600;
    color: rgba(255,255,255,0.7);
  }

  .drop-hint {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.5);
  }

  .file-selected {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    justify-content: center;
  }

  .check {
    width: 48px;
    height: 48px;
    background: #00ff00;
    color: #000;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    font-weight: 900;
  }

  .file-name {
    font-size: 1.1rem;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 0.25rem;
  }

  .file-size {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.5);
  }

  .error-box {
    background: rgba(255,0,0,0.1);
    border-left: 3px solid #ff0000;
    padding: 1rem;
    margin-bottom: 2rem;
  }

  .error-box p {
    font-size: 1rem;
    line-height: 1.6;
    color: rgba(255,255,255,0.9);
  }

  .error-box strong {
    color: #ff0000;
  }

  .analyze-button {
    width: 100%;
    background: linear-gradient(135deg, #ff0000 0%, #cc0000 100%);
    color: #ffffff;
    border: none;
    padding: 1.25rem;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    cursor: pointer;
    font-family: inherit;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(255,0,0,0.3);
  }

  .analyze-button:hover:not(:disabled) {
    background: linear-gradient(135deg, #cc0000 0%, #990000 100%);
    transform: translateY(-2px);
    box-shadow: 0 6px 30px rgba(255,0,0,0.5);
  }

  .analyze-button:disabled {
    background: rgba(255,255,255,0.1);
    color: rgba(255,255,255,0.3);
    cursor: not-allowed;
    box-shadow: none;
  }

  .offline-note {
    margin-top: 2rem;
    padding: 1rem;
    background: rgba(255,255,255,0.05);
    border-left: 3px solid rgba(255,255,255,0.2);
  }

  .offline-note p {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.6);
  }

  /* RESULTS VIEW */
  .new-lap-btn {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.15);
    color: rgba(255,255,255,0.7);
    padding: 0.75rem 1.5rem;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    font-family: inherit;
    transition: all 0.2s;
  }

  .new-lap-btn:hover {
    background: rgba(255,255,255,0.1);
    color: #ffffff;
    border-color: #ff0000;
  }

  .new-lap-btn.big {
    width: 100%;
    padding: 1.25rem;
    font-size: 1.1rem;
    margin-top: 2rem;
  }

  .lap-summary {
    margin-bottom: 4rem;
  }

  .summary-intro {
    background: rgba(255,0,0,0.05);
    border-left: 3px solid #ff0000;
    padding: 1.5rem;
    margin-bottom: 2rem;
  }

  .summary-intro p {
    font-size: 1.15rem;
    line-height: 1.7;
    color: rgba(255,255,255,0.9);
  }

  .summary-intro strong {
    color: #ffffff;
    font-weight: 700;
  }

  .time-gain {
    color: #ff0000;
  }

  .time-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1.5rem;
  }

  .time-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 1.5rem;
  }

  .time-card.optimal {
    border-color: rgba(0,255,0,0.3);
    background: rgba(0,255,0,0.03);
  }

  .time-card.delta {
    border-color: rgba(255,0,0,0.3);
    background: rgba(255,0,0,0.03);
  }

  .time-card label {
    display: block;
    font-size: 0.9rem;
    color: rgba(255,255,255,0.6);
    margin-bottom: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
  }

  .time-big {
    font-size: 2.5rem;
    font-weight: 900;
    font-family: 'SF Mono', monospace;
    color: #ffffff;
    margin-bottom: 0.5rem;
  }

  .time-card.optimal .time-big {
    color: #00ff00;
  }

  .time-card.delta .time-big {
    color: #ff0000;
  }

  .time-note {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.5);
    line-height: 1.5;
  }

  .driver-profile,
  .performance-section,
  .improvements-section,
  .training-section,
  .skill-gaps-section,
  .next-steps {
    margin-bottom: 4rem;
  }

  .profile-box {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 2rem;
  }

  .archetype-badge {
    display: inline-block;
    background: linear-gradient(135deg, #ff0000 0%, #cc0000 100%);
    color: #ffffff;
    padding: 0.75rem 1.5rem;
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
  }

  .confidence-note {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.6);
    margin-bottom: 2rem;
  }

  .traits-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
  }

  .traits-col label {
    display: block;
    font-size: 0.9rem;
    color: rgba(255,255,255,0.6);
    margin-bottom: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
  }

  .trait {
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.95rem;
    line-height: 1.5;
  }

  .trait.strength {
    background: rgba(0,255,0,0.05);
    border-left: 3px solid #00ff00;
    color: rgba(255,255,255,0.9);
  }

  .trait.weakness {
    background: rgba(255,0,0,0.05);
    border-left: 3px solid #ff0000;
    color: rgba(255,255,255,0.9);
  }

  .metrics-list {
    display: flex;
    flex-direction: column;
    gap: 2rem;
  }

  .metric-row {
    display: grid;
    grid-template-columns: 2fr 3fr auto;
    gap: 2rem;
    align-items: center;
  }

  .metric-info label {
    display: block;
    font-size: 1.1rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.25rem;
  }

  .metric-info p {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.6);
    line-height: 1.5;
  }

  .metric-bar {
    height: 12px;
    background: rgba(255,255,255,0.1);
    position: relative;
    overflow: hidden;
  }

  .bar-fill {
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    background: linear-gradient(90deg, #ff0000 0%, #00ff00 100%);
    transition: width 0.8s ease;
  }

  .metric-score {
    font-size: 1.5rem;
    font-weight: 900;
    font-family: 'SF Mono', monospace;
    color: #ffffff;
    min-width: 80px;
    text-align: right;
  }

  .improvements-list {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .improvement-card {
    background: rgba(255,255,255,0.02);
    border-left: 3px solid #ff0000;
    padding: 1.5rem;
  }

  .improvement-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .priority-badge {
    background: rgba(255,0,0,0.2);
    color: #ff0000;
    padding: 0.5rem 0.75rem;
    font-size: 0.85rem;
    font-weight: 700;
    font-family: 'SF Mono', monospace;
  }

  .gain {
    font-size: 1.25rem;
    font-weight: 900;
    font-family: 'SF Mono', monospace;
    color: #ff0000;
  }

  .improvement-text {
    font-size: 1.05rem;
    line-height: 1.7;
    color: rgba(255,255,255,0.9);
  }

  .drills-list {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  .drill-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 1.5rem;
  }

  .drill-number {
    font-size: 0.85rem;
    color: #ff0000;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1rem;
  }

  .drill-content {
    font-size: 1.05rem;
    line-height: 1.8;
    color: rgba(255,255,255,0.9);
    white-space: pre-wrap;
  }

  .gaps-list {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .gap-item {
    background: rgba(255,0,0,0.05);
    border-left: 3px solid #ff0000;
    padding: 1rem 1.5rem;
    font-size: 1.05rem;
    line-height: 1.6;
    color: rgba(255,255,255,0.9);
  }

  .steps-box {
    background: rgba(255,255,255,0.02);
    border-left: 3px solid rgba(255,255,255,0.15);
    padding: 1.5rem;
    margin-bottom: 2rem;
  }

  .steps-box p {
    font-size: 1.05rem;
    line-height: 1.7;
    color: rgba(255,255,255,0.85);
    margin-bottom: 1rem;
  }

  .steps-box p:last-child {
    margin-bottom: 0;
  }

  .visualizations-section {
    margin: 4rem 0;
  }

  .viz-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 1.5rem;
    margin-bottom: 1.5rem;
  }

  @media (max-width: 768px) {
    .viz-grid {
      grid-template-columns: 1fr;
    }
    .page {
      padding: 2rem 1.5rem;
    }

    h1 {
      font-size: 1.5rem;
    }

    h2 {
      font-size: 1.25rem;
    }

    .time-cards {
      grid-template-columns: 1fr;
    }

    .traits-grid {
      grid-template-columns: 1fr;
    }

    .metric-row {
      grid-template-columns: 1fr;
      gap: 1rem;
    }

    .metric-score {
      text-align: left;
    }

    .header {
      flex-direction: column;
      align-items: flex-start;
      gap: 1rem;
    }
  }
</style>
