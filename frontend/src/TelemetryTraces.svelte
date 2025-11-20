<script>
  export let arcLength = [];
  export let throttle = [];
  export let brakeFront = [];
  export let brakeRear = [];
  export let title = "BRAKE & THROTTLE TRACES";

  let svgWidth = 800;
  let svgHeight = 300;
  let padding = {top: 20, right: 40, bottom: 40, left: 60};

  $: xMin = 0;
  $: xMax = arcLength.length > 0 ? Math.max(...arcLength) : 1000;
  $: throttleMax = 100;
  $: brakeMax = brakeFront.length > 0 ? Math.max(Math.max(...brakeFront), Math.max(...brakeRear)) * 1.1 : 100;

  function scaleX(x) {
    return padding.left + (x / xMax) * (svgWidth - padding.left - padding.right);
  }

  function scaleYThrottle(y) {
    return svgHeight - padding.bottom - (y / throttleMax) * (svgHeight - padding.top - padding.bottom);
  }

  function scaleYBrake(y) {
    return svgHeight - padding.bottom - (y / brakeMax) * (svgHeight - padding.top - padding.bottom);
  }

  $: throttlePath = arcLength.length > 0 && throttle.length > 0
    ? `M ${arcLength.map((arc, i) => `${scaleX(arc)},${scaleYThrottle(throttle[i])}`).join(' L ')}`
    : '';

  $: brakeFrontPath = arcLength.length > 0 && brakeFront.length > 0
    ? `M ${arcLength.map((arc, i) => `${scaleX(arc)},${scaleYBrake(brakeFront[i])}`).join(' L ')}`
    : '';

  $: brakeRearPath = arcLength.length > 0 && brakeRear.length > 0
    ? `M ${arcLength.map((arc, i) => `${scaleX(arc)},${scaleYBrake(brakeRear[i])}`).join(' L ')}`
    : '';

  $: xGridLines = Array.from({length: 6}, (_, i) => (xMax / 5) * i);
</script>

<div class="telemetry-traces-container">
  <div class="telemetry-traces-header">
    <h3>{title}</h3>
  </div>

  {#if arcLength.length > 0}
    <svg width={svgWidth} height={svgHeight} class="telemetry-svg">
      <!-- X-axis grid lines -->
      {#each xGridLines as xVal}
        <line
          x1={scaleX(xVal)}
          y1={padding.top}
          x2={scaleX(xVal)}
          y2={svgHeight - padding.bottom}
          stroke="rgba(255,255,255,0.05)"
          stroke-width="1"
        />
        <text
          x={scaleX(xVal)}
          y={svgHeight - padding.bottom + 25}
          text-anchor="middle"
          class="axis-label"
        >
          {Math.round(xVal)}m
        </text>
      {/each}

      <!-- Y-axis labels -->
      <text
        x={padding.left - 10}
        y={padding.top + 10}
        text-anchor="end"
        class="axis-label"
      >
        100%
      </text>
      <text
        x={padding.left - 10}
        y={svgHeight - padding.bottom}
        text-anchor="end"
        class="axis-label"
      >
        0
      </text>

      <!-- Brake front trace (red) -->
      {#if brakeFrontPath}
        <path
          d={brakeFrontPath}
          fill="none"
          stroke="#ff0000"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          opacity="0.8"
        />
      {/if}

      <!-- Brake rear trace (orange) -->
      {#if brakeRearPath}
        <path
          d={brakeRearPath}
          fill="none"
          stroke="#ff6600"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          opacity="0.6"
        />
      {/if}

      <!-- Throttle trace (green) -->
      {#if throttlePath}
        <path
          d={throttlePath}
          fill="none"
          stroke="#00ff00"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          opacity="0.7"
        />
      {/if}

      <!-- Axes -->
      <line
        x1={padding.left}
        y1={svgHeight - padding.bottom}
        x2={svgWidth - padding.right}
        y2={svgHeight - padding.bottom}
        stroke="rgba(255,255,255,0.3)"
        stroke-width="2"
      />
      <line
        x1={padding.left}
        y1={padding.top}
        x2={padding.left}
        y2={svgHeight - padding.bottom}
        stroke="rgba(255,255,255,0.3)"
        stroke-width="2"
      />
    </svg>

    <div class="trace-legend">
      <div class="legend-item">
        <div class="legend-line brake-front"></div>
        <span>Brake Front</span>
      </div>
      <div class="legend-item">
        <div class="legend-line brake-rear"></div>
        <span>Brake Rear</span>
      </div>
      <div class="legend-item">
        <div class="legend-line throttle"></div>
        <span>Throttle</span>
      </div>
    </div>

    <div class="trace-stats">
      <div class="stat-item">
        <span class="stat-label">Peak Brake (Front):</span>
        <span class="stat-value">{Math.max(...brakeFront).toFixed(1)} bar</span>
      </div>
      <div class="stat-item">
        <span class="stat-label">Peak Brake (Rear):</span>
        <span class="stat-value">{Math.max(...brakeRear).toFixed(1)} bar</span>
      </div>
      <div class="stat-item">
        <span class="stat-label">Full Throttle %:</span>
        <span class="stat-value">{((throttle.filter(t => t > 95).length / throttle.length) * 100).toFixed(1)}%</span>
      </div>
    </div>
  {:else}
    <div class="no-data">
      <p>No telemetry data available</p>
    </div>
  {/if}
</div>

<style>
  .telemetry-traces-container {
    background: rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 4px;
    padding: 1.5rem;
    margin: 1.5rem 0;
  }

  .telemetry-traces-header {
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 2px solid #ff0000;
  }

  .telemetry-traces-header h3 {
    font-size: 1.1rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    letter-spacing: 0.05em;
  }

  .telemetry-svg {
    width: 100%;
    height: auto;
    background: rgba(0,0,0,0.6);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 2px;
  }

  .axis-label {
    fill: rgba(255,255,255,0.5);
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
  }

  .trace-legend {
    display: flex;
    gap: 2rem;
    margin-top: 1rem;
    flex-wrap: wrap;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.9rem;
    color: rgba(255,255,255,0.7);
  }

  .legend-line {
    width: 30px;
    height: 3px;
  }

  .legend-line.brake-front {
    background: #ff0000;
  }

  .legend-line.brake-rear {
    background: #ff6600;
  }

  .legend-line.throttle {
    background: #00ff00;
  }

  .trace-stats {
    display: flex;
    gap: 2rem;
    margin-top: 1rem;
    flex-wrap: wrap;
  }

  .stat-item {
    display: flex;
    gap: 0.5rem;
    font-size: 0.9rem;
  }

  .stat-label {
    color: rgba(255,255,255,0.6);
  }

  .stat-value {
    color: #ff0000;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
  }

  .no-data {
    padding: 3rem;
    text-align: center;
    color: rgba(255,255,255,0.4);
    font-style: italic;
  }

  @media (max-width: 768px) {
    .telemetry-traces-container {
      padding: 1rem;
    }

    .trace-legend, .trace-stats {
      flex-direction: column;
      gap: 0.5rem;
    }
  }
</style>
