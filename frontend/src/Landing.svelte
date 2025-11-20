<script>
  export let onEnter;
</script>

<div class="page">
  <div class="content">
    <div class="header">
      <h1>GR GARAGE COACH</h1>
      <div class="tagline">Everyone should get their shot at the track</div>
    </div>

    <section class="story">
      <p class="lead">
        Back in 2007, Akio Toyoda had a simple idea. He wanted to go racing, so he started a team out of his garage.
        Not as the president of Toyota, just as someone who genuinely loved driving. That garage became Gazoo Racing,
        and it was built around a beautiful concept: driving on real roads makes you a better driver,
        and better drivers make better cars. They call it kaizen. Continuous improvement through actually doing the thing.
      </p>

      <p class="lead">
        Fast forward to today, and we have GR Cup. Everyone runs the same GR86. Same engine, same tires, same
        everything. It's designed so that skill wins, not budget. And that's genuinely cool.
      </p>

      <p class="lead">
        But there's one piece that's still out of reach for a lot of talented drivers. Professional data coaching
        can run $5,000 to $15,000 per season. The cars might be equal, but access to getting better? That part's
        still pretty uneven. And we thought, why should it be?
      </p>
    </section>

    <section class="section">
      <h2>What we're trying to do here</h2>

      <p>
        We built a free telemetry coaching tool that gives you the same kind of analysis you'd get from a professional
        data coach. Upload your lap data and get back detailed feedback on where you're losing time, corner by corner,
        meter by meter. Completely free to use.
      </p>

      <div class="insight">
        <p>
          TRD already releases championship telemetry data from actual race weekends: GPS coordinates,
          brake pressure, throttle position, g forces, all of it. We take the fastest 10% of laps from real GR Cup races
          and use those as the baseline. Then we compare your lap to theirs and show you exactly where the differences are.
        </p>
      </div>

      <p>
        When we tell you "Turn 3: brake 15 meters later," you get <em>fifteen meters</em> at a
        <em>specific GPS coordinate</em>. Your apex speed is 87 km/h, the quick laps are hitting 94 km/h, and your
        traction circle shows you're using 74% of the available grip. Your missing 7 km/h is hiding right there.
      </p>
    </section>

    <section class="section">
      <h2>A bit about the cost</h2>

      <div class="cost-grid">
        <div class="cost-item">
          <div class="cost-number">$75 to 100</div>
          <div class="cost-label">per hour</div>
          <p>What a professional data coach typically charges to review your telemetry and walk you through it.</p>
        </div>

        <div class="cost-item">
          <div class="cost-number">$5,000 to 15,000</div>
          <div class="cost-label">per season</div>
          <p>Full season coaching with regular data analysis. It's genuinely helpful, just out of reach for most.</p>
        </div>

        <div class="cost-item highlight">
          <div class="cost-number">FREE</div>
          <div class="cost-label">always</div>
          <p>What this costs. Upload your CSV, get your analysis, get faster.</p>
        </div>
      </div>

      <div class="reality-check">
        <p>
          Look, motorsport has always had a barrier to entry problem. Most professional drivers came from families
          with the resources to support them. The talent pool should be bigger than it is. There are fast drivers
          out there who just never got the chance to prove it.
        </p>
        <p>
          GR Cup is already moving in the right direction. Everyone runs the same car, entry costs are more reasonable,
          and talent actually has a shot. They've built something really good. We just wanted to add one more piece:
          making quality coaching accessible to everyone, regardless of budget.
        </p>
      </div>
    </section>

    <section class="section">
      <h2>How it actually works</h2>

      <div class="how-section">
        <h3>Real championship data</h3>
        <p>
          We use the actual GR Cup race data from the TRD dataset. All seven tracks: Road America, COTA, Barber,
          Indianapolis, Sebring, Sonoma, and VIR. We pull the fastest 10% of laps and analyze every meter of
          every corner. That becomes the baseline we compare your lap against.
        </p>
        <p>
          The technical term is arc length parameterization, but basically it lets us line up your lap with the
          fast laps meter for meter, regardless of how different your speeds are. So when we see that the fast laps
          are braking at the 1,450 meter mark and you're braking at 1,465 meters, we can tell you exactly where
          you're giving up time and how much.
        </p>
      </div>

      <div class="how-section">
        <h3>Real physics, not guesswork</h3>
        <p>
          We calculate track curvature at every point using GPS coordinates, which tells us the theoretical maximum
          speed through each corner based on what a GR86 can actually do. The car weighs 1,270 kg with driver,
          the tires have a coefficient of about 1.35, maximum brake deceleration is around 1.4g. All real numbers.
        </p>
        <p>
          We account for weight transfer (which cuts your lateral grip by about 8% when braking), aero drag,
          tire slip angles, the whole thing. It's the same vehicle dynamics that race engineers use, just
          made accessible.
        </p>
        <p class="technical-note">
          For the curious: κ(s) = |x'y'' − y'x''| / (x'² + y'²)^(3/2)<br/>
          Maximum corner speed: v = √(a_lat_max / κ)<br/>
          Same math the pros use. Now you have it too.
        </p>
      </div>

      <div class="how-section">
        <h3>Actionable coaching</h3>
        <p>
          Instead of vague feedback like "you're slow in Turn 3," here's the kind of thing you'll get:
        </p>
        <div class="example-coaching">
          <p>
            "Turn 3 (arc 1450 to 1580m): You're braking at 1465m. The fast laps are braking at 1450m.
            That's costing you roughly 0.38 seconds (we estimate between 0.32 and 0.44).
          </p>
          <p>
            Your apex speed is 87 km/h, they're hitting 94 km/h. Your traction circle shows 0.74 when
            you could be at 0.93. You're using about 74% of the available grip. There's room to carry
            7 more km/h through the middle of the corner.
          </p>
          <p>
            Progressive braking practice: Run three laps braking at 1,455m instead
            of 1,465m. Then move it to 1,452m for three more. Then 1,450m. Try adding 1 km/h to your
            apex speed each lap until you reach 94 km/h or 0.93 on the traction circle."
          </p>
        </div>
        <p>
          GPS positions you can actually find on track. Speed targets you can work toward. Traction numbers
          you can measure. Practice drills you can go run this weekend.
        </p>
      </div>

      <div class="how-section">
        <h3>Understanding your driving style</h3>
        <p>
          We analyze 50+ measurements from your telemetry to understand your natural driving patterns, whether
          you tend to brake late, use smooth inputs, or have an aggressive style. It's based on what the data
          actually shows, not on what you think you're doing.
        </p>
        <p>
          Different styles need different approaches to improvement. Someone who brakes late will benefit from
          different practice drills than someone who's smooth but a bit timid. We try to match the coaching
          to how you naturally drive.
        </p>
      </div>
    </section>

    <section class="section">
      <h2>The philosophy behind this</h2>

      <p>
        Gazoo Racing's whole philosophy is kaizen. Continuous improvement through actually doing the thing.
        Learn from the track, bring it back to the garage, get better, repeat. It's a beautiful way to
        approach racing.
      </p>

      <p>
        But you can't improve what you can't measure. And until recently, getting meaningful measurements meant
        either having a coach who costs $75+ an hour, or trying to figure it all out yourself from raw data.
      </p>

      <p>
        GR Cup has done something really special. One of the most accessible paths into professional racing.
        Same cars, reasonable costs, genuine skill competition. We wanted to add one more piece:
        coaching available to everyone.
      </p>

      <p>
        Upload your telemetry, see where you're losing time, get specific suggestions on how to improve.
        Practice those drills. Get faster. Keep improving.
      </p>
    </section>

    <section class="section subtle">
      <h2>A few things worth mentioning</h2>

      <p>
        Everything comes from real championship telemetry or measured vehicle specs. Only actual data,
        measured numbers, and specific advice you can act on.
      </p>

      <p>
        We don't store your data. The analysis happens while you're using the site. Your
        telemetry stays in your browser and gets cleared when you leave. We never see it or save it.
      </p>

      <p>
        Just upload your TRD CSV file and you'll get your coaching. Free to use, always.
      </p>
    </section>

    <div class="enter-section">
      <button class="enter-button" on:click={onEnter}>
        ANALYZE MY LAP
      </button>
      <div class="enter-note">
        Upload telemetry / Get coaching / Get faster
      </div>
    </div>

    <div class="footer-note">
      Built for TRD Hackathon 2025 using real GR Cup championship data
    </div>
  </div>
</div>

<style>
  .page {
    background: #000000;
    color: #e0e0e0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'SF Pro Text', 'Helvetica Neue', sans-serif;
    min-height: 100vh;
    padding: 3rem 2rem 4rem;
    position: relative;
    overflow-x: hidden;
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
    max-width: 780px;
    margin: 0 auto;
    position: relative;
    z-index: 2;
  }

  .header {
    margin-bottom: 3rem;
    padding-bottom: 1.5rem;
    border-bottom: 2px solid #ff0000;
  }

  h1 {
    font-size: 2.5rem;
    font-weight: 800;
    margin: 0 0 0.5rem 0;
    color: #ffffff;
    letter-spacing: 0.02em;
  }

  h1::first-letter {
    color: #ff0000;
  }

  .tagline {
    font-size: 1.15rem;
    color: rgba(255,255,255,0.6);
    font-weight: 400;
  }

  .story {
    margin-bottom: 4rem;
  }

  .lead {
    font-size: 1.2rem;
    line-height: 1.8;
    color: rgba(255,255,255,0.9);
    margin-bottom: 1.5rem;
    font-weight: 400;
  }

  .section {
    margin-bottom: 4rem;
  }

  .section.subtle {
    opacity: 0.8;
  }

  h2 {
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0 0 1.5rem 0;
    color: #ffffff;
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

  h3 {
    font-size: 1.2rem;
    font-weight: 600;
    margin: 0 0 1rem 0;
    color: #ff0000;
  }

  p {
    font-size: 1.05rem;
    line-height: 1.75;
    color: rgba(255,255,255,0.85);
    margin-bottom: 1.25rem;
  }

  em {
    color: #ff0000;
    font-style: normal;
    font-weight: 600;
  }

  .insight {
    background: rgba(255,0,0,0.05);
    border-left: 3px solid #ff0000;
    padding: 1.5rem;
    margin: 2rem 0;
  }

  .insight p {
    margin: 0;
    font-size: 1.1rem;
    line-height: 1.7;
  }

  .cost-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
  }

  .cost-item {
    padding: 1.5rem;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.1);
  }

  .cost-item.highlight {
    background: rgba(255,0,0,0.08);
    border-color: rgba(255,0,0,0.4);
  }

  .cost-number {
    font-size: 2rem;
    font-weight: 800;
    color: rgba(255,255,255,0.5);
    margin-bottom: 0.25rem;
    font-family: 'SF Mono', monospace;
  }

  .cost-item.highlight .cost-number {
    color: #00ff00;
  }

  .cost-label {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: rgba(255,255,255,0.4);
    margin-bottom: 1rem;
  }

  .cost-item p {
    font-size: 0.95rem;
    margin: 0;
    line-height: 1.6;
  }

  .reality-check {
    margin: 2.5rem 0;
    padding: 1.5rem;
    background: rgba(0,0,0,0.3);
    border-top: 1px solid rgba(255,255,255,0.1);
    border-bottom: 1px solid rgba(255,255,255,0.1);
  }

  .reality-check p {
    margin-bottom: 1rem;
  }

  .reality-check p:last-child {
    margin-bottom: 0;
  }

  .how-section {
    margin-bottom: 3rem;
  }

  .technical-note {
    font-family: 'SF Mono', 'Courier New', monospace;
    font-size: 0.9rem;
    color: rgba(255,255,255,0.6);
    background: rgba(255,255,255,0.02);
    padding: 1rem;
    margin-top: 1rem;
    line-height: 1.6;
  }

  .example-coaching {
    background: rgba(255,0,0,0.05);
    border-left: 3px solid #ff0000;
    padding: 1.5rem;
    margin: 1.5rem 0;
    font-family: 'SF Mono', 'Courier New', monospace;
    font-size: 0.95rem;
  }

  .example-coaching p {
    margin-bottom: 1rem;
    line-height: 1.7;
  }

  .example-coaching p:last-child {
    margin-bottom: 0;
  }

  .enter-section {
    margin: 4rem 0 2rem;
    text-align: center;
  }

  .enter-button {
    background: linear-gradient(135deg, #ff0000 0%, #cc0000 100%);
    color: #ffffff;
    border: none;
    padding: 1.25rem 3rem;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    cursor: pointer;
    font-family: inherit;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(255,0,0,0.3);
  }

  .enter-button:hover {
    background: linear-gradient(135deg, #cc0000 0%, #990000 100%);
    transform: translateY(-2px);
    box-shadow: 0 6px 30px rgba(255,0,0,0.5);
  }

  .enter-note {
    margin-top: 1rem;
    font-size: 0.95rem;
    color: rgba(255,255,255,0.5);
  }

  .footer-note {
    text-align: center;
    font-size: 0.85rem;
    color: rgba(255,255,255,0.4);
    margin-top: 3rem;
    padding-top: 2rem;
    border-top: 1px solid rgba(255,255,255,0.1);
  }

  @media (max-width: 768px) {
    .page {
      padding: 2rem 1.5rem 3rem;
    }

    h1 {
      font-size: 2rem;
    }

    .lead {
      font-size: 1.1rem;
    }

    h2 {
      font-size: 1.4rem;
    }

    .cost-grid {
      grid-template-columns: 1fr;
    }

    .enter-button {
      width: 100%;
      padding: 1rem 2rem;
    }
  }
</style>
