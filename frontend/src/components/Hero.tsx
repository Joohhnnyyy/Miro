

export const Hero = () => {
  return (
    <>
<canvas id="hero-kv" aria-hidden="true" width="1974" height="1540"></canvas>

<div className="pxctl" id="pxctl">
  <span className="lbl">Cell</span>
  <button data-cell="22">L</button><button data-cell="14">M</button><button data-cell="9" className="on">S</button>
  <span className="lbl">Brush</span>
  <button data-brush="16">L</button><button data-brush="10" className="on">M</button><button data-brush="7">S</button>
</div>

<a className="toplink" id="toplink" href="/detect" data-cursor-label="Detector">Launch AI Detector <svg className="ar" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M7 17 17 7"></path><path d="M8 7h9v9"></path></svg></a>


  <section className="hero" id="hero">
    <div className="hhead">
      <div className="hgrid">
        <h1 className="hl"><span className="ln"><span>Content,</span></span><span className="ln"><span>verified.</span></span></h1>
        <div className="hcol">
          <p className="hdesc"><span className="ln"><span>The advanced AI detection</span></span><span className="ln"><span>platform for modern writing</span></span></p>
          <div className="hfoot">
            <span className="wlogo"><img src="/assets/Miro_black_logo.png" alt="MIRO" width="56" height="auto" /></span>
            <p className="htag">Detect AI-generated text, analyze writing patterns, and verify human authenticity with precision.</p>
          </div>
        </div>
      </div>
    </div>
  </section>
    </>
  );
};
