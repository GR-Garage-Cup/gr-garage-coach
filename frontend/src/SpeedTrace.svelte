<script>
  export let arcLength = [];
  export let speed = [];
  export let title = "SPEED TRACE";

  let svgWidth = 800;
  let svgHeight = 250;
  let padding = {top: 20, right: 40, bottom: 40, left: 60};

  $: xMin = 0;
  $: xMax = arcLength.length > 0 ? Math.max(...arcLength) : 1000;
  $: yMin = 0;
  $: yMax = speed.length > 0 ? Math.max(...speed) * 1.1 : 200;

  function scaleX(x) {
    return padding.left + (x / xMax) * (svgWidth - padding.left - padding.right);
  }

  function scaleY(y) {
    return svgHeight - padding.bottom - (y / yMax) * (svgHeight - padding.top - padding.bottom);
  }

  $: pathData = arcLength.length > 0 && speed.length > 0
    ? `M ${arcLength.map((arc, i) => `${scaleX(arc)},${scaleY(speed[i])}`).join(' L ')}`
    : '';

  $: yGridLines = [0, 50, 100, 150, 200].filter(v => v <= yMax);
  $: xGridLines = Array.from({length: 6}, (_, i) => (xMax / 5) * i);
</script>

<div class="speed-trace-container">
  <div class="speed-trace-header">
    <h3>{title}</h3>
  </div>

  {#if arcLength.length > 0 && speed.length > 0}
    <svg width={svgWidth} height={svgHeight} class="speed-svg">
      <!-- Y-axis grid lines -->
      {#each yGridLines as yVal}
        <line
          x1={padding.left}
          y1={scaleY(yVal)}
          x2={svgWidth - padding.right}
          y2={scaleY(yVal)}
          stroke="rgba(255,255,255,0.05)"
          stroke-width="1"
        />
        <text
          x={padding.left - 10}
          y={scaleY(yVal)}
          text-anchor="end"
          dominant-baseline="middle"
          class="axis-label"
        >
          {yVal} km/h
        </text>
      {/each}

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

      <!-- Speed trace line -->
      <path
        d={pathData}
        fill="none"
        stroke="#ff0000"
        stroke-width="2.5"
        stroke-linecap="round"
        stroke-linejoin="round"
      />

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

    <div class="trace-stats">
      <div class="stat-item">
        <span class="stat-label">Peak Speed:</span>
        <span class="stat-value">{Math.max(...speed).toFixed(1)} km/h</span>
      </div>
      <div class="stat-item">
        <span class="stat-label">Avg Speed:</span>
        <span class="stat-value">{(speed.reduce((a,b) => a+b, 0) / speed.length).toFixed(1)} km/h</span>
      </div>
      <div class="stat-item">
        <span class="stat-label">Distance:</span>
        <span class="stat-value">{Math.max(...arcLength).toFixed(0)}m</span>
      </div>
    </div>
  {:else}
    <div class="no-data">
      <p>No speed data available</p>
    </div>
  {/if}
</div>

<style>
  .speed-trace-container {
    background: rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 4px;
    padding: 1.5rem;
    margin: 1.5rem 0;
  }

  .speed-trace-header {
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 2px solid #ff0000;
  }

  .speed-trace-header h3 {
    font-size: 1.1rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    letter-spacing: 0.05em;
  }

  .speed-svg {
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
    .speed-trace-container {
      padding: 1rem;
    }

    .trace-stats {
      flex-direction: column;
      gap: 0.5rem;
    }
  }
</style>
