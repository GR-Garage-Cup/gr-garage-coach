<script>
  export let longitudinalG = [];
  export let lateralG = [];
  export let title = "TRACTION CIRCLE";

  let svgSize = 400;
  let padding = 40;
  let centerX = svgSize / 2;
  let centerY = svgSize / 2;
  let radius = (svgSize - 2 * padding) / 2;

  let maxG = 1.5;

  function scaleG(gValue) {
    return (gValue / maxG) * radius;
  }

  function toSVG(longG, latG) {
    const x = centerX + scaleG(latG);
    const y = centerY - scaleG(longG);
    return {x, y};
  }

  $: points = longitudinalG.length > 0 && lateralG.length > 0
    ? longitudinalG.map((longG, i) => toSVG(longG, lateralG[i]))
    : [];

  $: circles = [0.5, 1.0, 1.5].map(g => scaleG(g));
</script>

<div class="traction-circle-container">
  <div class="traction-circle-header">
    <h3>{title}</h3>
  </div>

  {#if points.length > 0}
    <svg width={svgSize} height={svgSize} class="traction-svg">
      <!-- Reference circles -->
      {#each circles as r, i}
        <circle
          cx={centerX}
          cy={centerY}
          r={r}
          fill="none"
          stroke="rgba(255,255,255,{0.1 - i * 0.02})"
          stroke-width="1"
          stroke-dasharray={i === 2 ? "none" : "4,4"}
        />
        <text
          x={centerX + r - 30}
          y={centerY - 5}
          class="circle-label"
        >
          {(i + 1) * 0.5}G
        </text>
      {/each}

      <!-- Axes -->
      <line
        x1={centerX}
        y1={padding}
        x2={centerX}
        y2={svgSize - padding}
        stroke="rgba(255,255,255,0.2)"
        stroke-width="2"
      />
      <line
        x1={padding}
        y1={centerY}
        x2={svgSize - padding}
        y2={centerY}
        stroke="rgba(255,255,255,0.2)"
        stroke-width="2"
      />

      <!-- Axis labels -->
      <text x={centerX} y={20} text-anchor="middle" class="axis-label-text">Braking</text>
      <text x={centerX} y={svgSize - 10} text-anchor="middle" class="axis-label-text">Accel</text>
      <text x={10} y={centerY} text-anchor="start" class="axis-label-text">Left</text>
      <text x={svgSize - 10} y={centerY} text-anchor="end" class="axis-label-text">Right</text>

      <!-- Data points -->
      {#each points as point}
        <circle
          cx={point.x}
          cy={point.y}
          r="1.5"
          fill="#ff0000"
          opacity="0.4"
        />
      {/each}
    </svg>

    <div class="circle-stats">
      <div class="stat-item">
        <span class="stat-label">Peak Braking:</span>
        <span class="stat-value">{Math.abs(Math.min(...longitudinalG)).toFixed(2)}G</span>
      </div>
      <div class="stat-item">
        <span class="stat-label">Peak Lateral:</span>
        <span class="stat-value">{Math.max(...lateralG.map(Math.abs)).toFixed(2)}G</span>
      </div>
      <div class="stat-item">
        <span class="stat-label">Data Points:</span>
        <span class="stat-value">{points.length.toLocaleString()}</span>
      </div>
    </div>
  {:else}
    <div class="no-data">
      <p>No G-force data available</p>
    </div>
  {/if}
</div>

<style>
  .traction-circle-container {
    background: rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 4px;
    padding: 1.5rem;
    margin: 1.5rem 0;
  }

  .traction-circle-header {
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 2px solid #ff0000;
  }

  .traction-circle-header h3 {
    font-size: 1.1rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    letter-spacing: 0.05em;
  }

  .traction-svg {
    width: 100%;
    height: auto;
    background: rgba(0,0,0,0.6);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 2px;
    display: block;
    margin: 0 auto;
  }

  .circle-label {
    fill: rgba(255,255,255,0.4);
    font-size: 10px;
    font-family: 'JetBrains Mono', monospace;
  }

  .axis-label-text {
    fill: rgba(255,255,255,0.6);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .circle-stats {
    display: flex;
    gap: 2rem;
    margin-top: 1rem;
    flex-wrap: wrap;
    justify-content: center;
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
    .traction-circle-container {
      padding: 1rem;
    }

    .circle-stats {
      flex-direction: column;
      gap: 0.5rem;
      align-items: center;
    }
  }
</style>
