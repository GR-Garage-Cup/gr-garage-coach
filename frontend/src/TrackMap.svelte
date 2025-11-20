<script>
  export let x = [];
  export let y = [];
  export let title = "TRACK MAP";

  let svgWidth = 600;
  let svgHeight = 400;
  let padding = 40;

  $: xMin = Math.min(...x);
  $: xMax = Math.max(...x);
  $: yMin = Math.min(...y);
  $: yMax = Math.max(...y);

  $: xRange = xMax - xMin;
  $: yRange = yMax - yMin;

  $: scale = Math.min(
    (svgWidth - 2 * padding) / xRange,
    (svgHeight - 2 * padding) / yRange
  );

  function toSVG(xCoord, yCoord) {
    const scaledX = (xCoord - xMin) * scale + padding;
    const scaledY = svgHeight - ((yCoord - yMin) * scale + padding);
    return `${scaledX},${scaledY}`;
  }

  $: pathData = x.length > 0 && y.length > 0
    ? `M ${x.map((xi, i) => toSVG(xi, y[i])).join(' L ')}`
    : '';
</script>

<div class="track-map-container">
  <div class="track-map-header">
    <h3>{title}</h3>
  </div>

  {#if x.length > 0 && y.length > 0}
    <svg width={svgWidth} height={svgHeight} class="track-svg">
      <!-- Grid lines -->
      <defs>
        <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
          <path d="M 50 0 L 0 0 0 50" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="1"/>
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill="url(#grid)" />

      <!-- Racing line -->
      <path
        d={pathData}
        fill="none"
        stroke="#ff0000"
        stroke-width="3"
        stroke-linecap="round"
        stroke-linejoin="round"
      />

      <!-- Start/finish marker -->
      <circle
        cx={toSVG(x[0], y[0]).split(',')[0]}
        cy={toSVG(x[0], y[0]).split(',')[1]}
        r="6"
        fill="#00ff00"
        stroke="#ffffff"
        stroke-width="2"
      />
    </svg>

    <div class="map-legend">
      <div class="legend-item">
        <div class="legend-marker start"></div>
        <span>Start/Finish</span>
      </div>
      <div class="legend-item">
        <div class="legend-marker line"></div>
        <span>Racing Line ({x.length} GPS points)</span>
      </div>
    </div>
  {:else}
    <div class="no-data">
      <p>No GPS data available</p>
    </div>
  {/if}
</div>

<style>
  .track-map-container {
    background: rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 4px;
    padding: 1.5rem;
    margin: 1.5rem 0;
  }

  .track-map-header {
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 2px solid #ff0000;
  }

  .track-map-header h3 {
    font-size: 1.1rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    letter-spacing: 0.05em;
  }

  .track-svg {
    width: 100%;
    height: auto;
    background: rgba(0,0,0,0.6);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 2px;
  }

  .map-legend {
    display: flex;
    gap: 2rem;
    margin-top: 1rem;
    font-size: 0.9rem;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: rgba(255,255,255,0.7);
  }

  .legend-marker {
    width: 20px;
    height: 3px;
  }

  .legend-marker.start {
    width: 12px;
    height: 12px;
    background: #00ff00;
    border: 2px solid #ffffff;
    border-radius: 50%;
  }

  .legend-marker.line {
    background: #ff0000;
  }

  .no-data {
    padding: 3rem;
    text-align: center;
    color: rgba(255,255,255,0.4);
    font-style: italic;
  }

  @media (max-width: 768px) {
    .track-map-container {
      padding: 1rem;
    }

    .map-legend {
      flex-direction: column;
      gap: 0.75rem;
    }
  }
</style>
