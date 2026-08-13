import { useState, useEffect } from 'react';

interface SignalData {
  signal_a: any;
  signal_b: any;
  signal_c: any;
  signal_d: any;
}

interface FeatureContribution {
  feature: string;
  weight: number;
  value: number;
  contribution: number;
}

interface SentenceData {
  text: string;
  ai_probability: number;
  is_ai: boolean;
  band: string;
}

interface SentenceDistribution {
  human_ratio: number;
  ai_ratio: number;
  human_count: number;
  ai_count: number;
  uncertain_count: number;
}

interface DetectResponse {
  is_ai: boolean;
  ai_probability: number;
  authenticity_score: number;
  sentence_distribution: SentenceDistribution;
  band?: string;
  band_label?: string;
  signals: SignalData;
  plain_english_explanation?: string;
  feature_contributions?: FeatureContribution[];
  sentences?: SentenceData[];
}

type ViewState = 'input' | 'scanning' | 'results';

export const Detector = () => {
  const [text, setText] = useState('');
  const [viewState, setViewState] = useState<ViewState>('input');
  const [result, setResult] = useState<DetectResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [scanProgress, setScanProgress] = useState(0);
  const [isHowItWorksOpen, setIsHowItWorksOpen] = useState(false);
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    window.hasNavigatedAwayFromHome = true;
  }, []);

  const wordCount = text.trim().split(/\s+/).filter(w => w.length > 0).length;
  const charCount = text.length;

  const analyzeEssay = async () => {
    if (wordCount < 10) {
      setError("Please enter a longer essay for accurate analysis.");
      return;
    }
    
    setError(null);
    setViewState('scanning');
    setScanProgress(0);

    // Simulate scanning progress
    const progressInterval = setInterval(() => {
      setScanProgress(prev => Math.min(prev + 5, 95));
    }, 100);

    try {
      const response = await fetch('/api/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text })
      });

      if (!response.ok) {
        throw new Error("Server error or analysis failed.");
      }

      const data = await response.json();
      
      // Ensure we show at least a tiny bit of the scanning animation
      setTimeout(() => {
        clearInterval(progressInterval);
        setScanProgress(100);
        setTimeout(() => {
          setResult(data);
          setViewState('results');
        }, 500);
      }, 1000);

    } catch (err: any) {
      clearInterval(progressInterval);
      setError("Error: " + err.message);
      setViewState('input');
    }
  };

  const Sidebar = () => (
    <aside className="hidden w-60 shrink-0 flex-col border-r border-border bg-background lg:flex">
      <div className="flex min-h-16 items-center justify-between border-b border-border px-5">
        <a href="/" className="flex items-center gap-3 transition-transform duration-300 ease-out hover:scale-[1.02] hover:opacity-80">
          <img src="/assets/Miro_black_logo.png" alt="MIRO Logo" className="h-10 w-auto" />
        </a>
        <span className="font-mono text-[10px] text-muted-foreground">v1.0</span>
      </div>
      <nav className="flex flex-1 flex-col p-3 text-[13px]">
        <p className="px-2 pb-2 pt-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">Workspace</p>
        
        <button onClick={() => { setViewState('input'); setText(''); setResult(null); }} className={`flex min-h-11 items-center gap-3 px-3 w-full text-left font-semibold transition-all duration-300 ease-out hover:scale-[1.02] hover:shadow-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring ${viewState === 'input' ? 'border border-primary bg-primary text-primary-foreground' : 'text-muted-foreground hover:bg-secondary hover:text-foreground'}`}>
          {/* @ts-ignore */}
          <iconify-icon icon="ph:plus-light" class="text-base"></iconify-icon>
          New analysis
        </button>

        {viewState === 'scanning' && (
          <div className="mt-1 flex min-h-11 items-center gap-3 border border-primary bg-primary px-3 font-semibold text-primary-foreground focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring">
            {/* @ts-ignore */}
            <iconify-icon icon="ph:spinner-light" class="text-base animate-spin"></iconify-icon>
            Scanning now
          </div>
        )}

        {viewState === 'results' && (
          <div className="mt-1 flex min-h-11 items-center gap-3 border border-primary bg-primary px-3 font-semibold text-primary-foreground focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring">
            {/* @ts-ignore */}
            <iconify-icon icon="ph:file-text-light" class="text-base"></iconify-icon>
            Evidence report
          </div>
        )}
        
        <button className="mt-1 flex min-h-11 items-center gap-3 px-3 w-full text-left text-muted-foreground transition-all duration-200 ease-out hover:translate-x-1 hover:bg-secondary hover:text-foreground focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring">
          {/* @ts-ignore */}
          <iconify-icon icon="ph:folder-light" class="text-base"></iconify-icon>
          Analysis archive
        </button>
        
        {viewState === 'input' && (
          <div className="mt-8 border-y border-border py-4">
            <p className="px-2 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">Signal guide</p>
            <div className="mt-3 space-y-3 px-2 text-[11px]">
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-2"><span className="h-2 w-2 bg-tertiary"></span>Human-led</span>
                <span className="font-mono text-muted-foreground">0–35</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-2"><span className="h-2 w-2 bg-accent"></span>Review</span>
                <span className="font-mono text-muted-foreground">36–70</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-2"><span className="h-2 w-2 bg-destructive"></span>High signal</span>
                <span className="font-mono text-muted-foreground">71–100</span>
              </div>
            </div>
          </div>
        )}

        <div className="mt-auto border-t border-border px-2 pt-4">
          <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">Academic integrity</p>
          <p className="mt-2 text-[11px] leading-5 text-muted-foreground">Evidence-led language analysis for editorial and academic decisions.</p>
        </div>
      </nav>
    </aside>
  );

  const renderHighlightedText = (text: string, highlight: string) => {
    if (!highlight.trim()) return text;
    const parts = text.split(new RegExp(`(${highlight})`, 'gi'));
    return (
      <>
        {parts.map((part, i) => 
          part.toLowerCase() === highlight.toLowerCase() 
            ? <mark key={i} className="bg-primary/40 text-foreground px-0.5 rounded-sm">{part}</mark>
            : part
        )}
      </>
    );
  };

  return (
    <div className="min-h-screen w-full bg-background flex flex-col relative font-sans">
      <div className="flex flex-1">
        <Sidebar />
        
        <main className="flex min-w-0 flex-1 flex-col">
          
          {/* Header */}
          <header className="flex min-h-16 items-center justify-between border-b border-border px-4 md:px-8">
            <div className="flex items-center gap-3">
              <button className="flex min-h-10 min-w-10 items-center justify-center border border-border lg:hidden focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring">
                {/* @ts-ignore */}
                <iconify-icon icon="ph:list-light" class="text-lg"></iconify-icon>
              </button>
              
              {viewState === 'input' && (
                <div className="flex items-center gap-2">
                  <span className="font-heading text-base font-semibold tracking-tight lg:hidden">MIRO</span>
                  <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">New analysis / 01</span>
                </div>
              )}
              
              {viewState === 'scanning' && (
                <div>
                  <p className="font-mono text-[10px] uppercase tracking-widest text-foreground">Analysis in progress</p>
                  <p className="mt-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">Deep scan</p>
                </div>
              )}

              {viewState === 'results' && (
                <div>
                  <span className="font-heading text-base font-semibold tracking-tight lg:hidden">MIRO</span>
                  <p className="mt-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground lg:hidden">Evidence report / 01</p>
                  <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground hidden lg:block">Evidence report / 01</p>
                </div>
              )}
            </div>

            {viewState === 'input' && (
              <button onClick={() => setIsHowItWorksOpen(true)} className="min-h-10 border border-border px-3 text-[13px] font-medium hover:bg-secondary active:bg-muted focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring">
                {/* @ts-ignore */}
                <iconify-icon icon="ph:question-light" class="mr-1 align-middle text-base"></iconify-icon>
                How it works
              </button>
            )}

            {viewState === 'scanning' && (
              <button onClick={() => setViewState('input')} className="min-h-10 border border-border px-3 text-[13px] font-medium hover:bg-secondary focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring">
                Cancel scan
              </button>
            )}

            {viewState === 'results' && (
              <div className="flex gap-2">
                <button onClick={() => { setViewState('input'); setText(''); setResult(null); }} className="min-h-10 border border-primary bg-primary px-3 text-[13px] font-semibold text-primary-foreground">
                  {/* @ts-ignore */}
                  <iconify-icon icon="ph:plus-light" class="mr-1 align-middle"></iconify-icon>
                  New analysis
                </button>
              </div>
            )}
          </header>

          {/* Input State */}
          {viewState === 'input' && (
            <section className="mx-auto flex w-full max-w-7xl flex-1 flex-col px-4 py-6 md:px-8 md:py-10">
              <div className="max-w-3xl">
                <p className="font-mono text-[10px] uppercase tracking-widest text-tertiary">Essay intelligence</p>
                <h1 className="mt-3 font-heading text-2xl font-bold tracking-tight">Bring an essay into focus.</h1>
                <p className="mt-3 max-w-2xl text-[13px] leading-6 text-muted-foreground">Start with a document or draft. MIRO maps writing signals so you can make a more informed review decision.</p>
              </div>
              
              <div className="mt-8 flex flex-1 flex-col border border-border">
                <section className="flex min-h-80 flex-col p-5 lg:p-8">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">01 / Compose</p>
                      <h2 className="mt-2 font-heading text-lg font-semibold">Paste your essay</h2>
                      <p className="mt-2 text-[13px] leading-5 text-muted-foreground">For responses, drafts, and excerpts. We evaluate linguistic patterns, not author identity.</p>
                    </div>
                    <button onClick={() => setText('')} disabled={!text} className="min-h-10 px-2 text-[13px] font-medium text-muted-foreground hover:text-foreground focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring disabled:cursor-not-allowed disabled:opacity-40">
                      Clear
                    </button>
                  </div>
                  
                  <label htmlFor="essay" className="mt-6 block font-mono text-[10px] font-semibold uppercase tracking-widest">Essay text</label>
                  <textarea 
                    id="essay" 
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    className="mt-2 min-h-64 w-full flex-1 resize-y border border-input bg-background p-4 text-base leading-relaxed text-foreground placeholder:text-muted-foreground transition-all duration-300 ease-out hover:border-foreground/50 hover:shadow-sm focus:border-primary focus:shadow-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring" 
                    placeholder="Enter the text you want to analyze here... It should be at least a few sentences long for the best results."
                  />
                  
                  <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
                    <p className="font-mono text-[10px] text-muted-foreground">{charCount} characters <span aria-hidden="true">·</span> {wordCount} words</p>
                    <p className="flex items-center gap-1 text-[11px] text-muted-foreground">
                      {/* @ts-ignore */}
                      <iconify-icon icon="ph:shield-check-light" class="text-base text-tertiary"></iconify-icon>
                      Private by design
                    </p>
                  </div>

                  {error && (
                    <div className="mt-4 border border-destructive bg-accent px-3 py-2 text-[11px] leading-4 text-foreground">
                      {/* @ts-ignore */}
                      <iconify-icon icon="ph:warning-circle-light" class="mr-1 align-middle text-destructive"></iconify-icon>
                      {error}
                    </div>
                  )}
                </section>
              </div>
              
              <footer className="flex flex-col gap-4 border-x border-b border-border bg-secondary p-5 md:flex-row md:items-center md:justify-between md:px-6">
                <div className="flex items-start gap-3">
                  <span className="mt-1 h-2 w-2 shrink-0 bg-tertiary"></span>
                  <p className="max-w-2xl text-[13px] leading-5 text-muted-foreground">Ready to analyze. Results include an overall signal, sentence-level evidence, and writing-quality metrics.</p>
                </div>
                <button onClick={analyzeEssay} className="group min-h-11 shrink-0 border border-primary bg-primary px-5 py-3 text-[13px] font-semibold text-primary-foreground transition-all duration-300 ease-out hover:bg-foreground hover:shadow-sm active:scale-[0.98] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring disabled:cursor-not-allowed disabled:opacity-40">
                  <span>Analyze essay</span>
                  {/* @ts-ignore */}
                  <iconify-icon icon="ph:arrow-right-light" class="ml-2 align-middle text-base transition-transform duration-300 group-hover:translate-x-0.5"></iconify-icon>
                </button>
              </footer>
              
              <div className="mt-4 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                <span>Secure workspace</span><span aria-hidden="true">/</span>
                <span>PDF · DOCX · TXT</span><span aria-hidden="true">/</span>
                <span>Keyboard first</span>
              </div>
            </section>
          )}

          {/* Scanning State */}
          {viewState === 'scanning' && (
            <section className="mx-auto flex w-full max-w-5xl flex-1 flex-col px-4 py-8 md:px-8 md:py-12">
              <div className="max-w-2xl">
                <h1 className="font-heading text-2xl font-bold tracking-tight">Reading the signals in your draft.</h1>
                <p className="mt-3 text-[13px] leading-6 text-muted-foreground">MIRO is segmenting the document, comparing linguistic variation, and assembling evidence for review.</p>
              </div>
              
              <section className="mt-8 border border-border bg-card">
                <div className="flex flex-col gap-6 border-b border-border p-5 md:flex-row md:items-center md:justify-between md:p-6">
                  <div className="flex items-center gap-4">
                    <span className="flex h-11 w-11 items-center justify-center border border-border bg-secondary">
                      {/* @ts-ignore */}
                      <iconify-icon icon="ph:scan-light" class="text-xl"></iconify-icon>
                    </span>
                    <div>
                      <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">Live scan</p>
                      <h2 className="mt-1 font-heading text-lg font-semibold">Deep scan in progress</h2>
                    </div>
                  </div>
                  <div className="md:text-right">
                    <p className="font-mono text-2xl font-bold">{scanProgress}%</p>
                    <p className="mt-1 text-[11px] text-muted-foreground">Analyzing structure</p>
                  </div>
                </div>
                
                <div className="p-5 md:p-6">
                  <div className="h-2 overflow-hidden bg-secondary">
                    <div className="h-full bg-tertiary transition-all duration-300 ease-out" style={{ width: `${scanProgress}%` }}></div>
                  </div>
                  
                  <div className="mt-8 border border-border bg-background p-5">
                    <div className="flex items-center justify-between border-b border-border pb-3">
                      <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">Document pass / 04</p>
                      <span className="flex items-center gap-2 text-[11px] text-foreground">
                        <span className="h-2 w-2 bg-tertiary animate-pulse"></span>Scanning
                      </span>
                    </div>
                    <div className="relative mt-5 overflow-hidden">
                      <div className="absolute inset-x-0 top-1/2 h-px bg-tertiary opacity-50"></div>
                      <div className="space-y-4">
                        <div className="h-3 w-full bg-secondary"></div>
                        <div className="h-3 w-11/12 bg-secondary"></div>
                        <div className="h-3 w-10/12 bg-secondary"></div>
                        <div className="h-3 w-full bg-secondary"></div>
                        <div className="h-3 w-9/12 bg-secondary"></div>
                        <div className="h-3 w-10/12 bg-secondary"></div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="mt-6 grid grid-cols-1 border border-border sm:grid-cols-3">
                    <div className="border-b border-border p-4 sm:border-b-0 sm:border-r">
                      <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">01 / Segmenting</p>
                      <p className="mt-2 text-[13px] font-semibold">{scanProgress > 10 ? 'Essay structure mapped' : 'Processing...'}</p>
                    </div>
                    <div className="border-b border-border p-4 sm:border-b-0 sm:border-r">
                      <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">02 / Comparing</p>
                      <p className="mt-2 text-[13px] font-semibold">{scanProgress > 50 ? 'Variation mapped' : 'Variation in progress'}</p>
                    </div>
                    <div className="bg-secondary p-4">
                      <p className="font-mono text-[10px] uppercase tracking-widest text-secondary-foreground">03 / Explaining</p>
                      <p className="mt-2 text-[13px] font-semibold">{scanProgress > 90 ? 'Evidence queued' : 'Waiting...'}</p>
                    </div>
                  </div>
                </div>
              </section>
            </section>
          )}

          {/* Results State */}
          {viewState === 'results' && result && (
            <>
            <section className="flex flex-1 flex-col px-4 py-6 md:px-8 pb-44 lg:pb-6">
              <div className="flex flex-col justify-between gap-4 border-b border-border pb-6 md:flex-row md:items-end">
                <div>
                  <h1 className="font-heading text-2xl font-bold tracking-tight">Essay Analysis Report</h1>
                  <p className="mt-2 text-[13px] text-muted-foreground">
                    {wordCount} words · {result.sentences?.length || 0} sentences assessed · Completed just now
                  </p>
                </div>
                <div className="hidden md:block">
                  <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground text-right">Analysis Complete</p>
                </div>
              </div>
              
              <div className="mt-6 grid min-w-0 flex-1 grid-cols-1 gap-6 lg:grid-cols-12">
                
                {/* Document Reader */}
                <article className="border border-border bg-card lg:col-span-7">
                  <div className="flex items-center justify-between border-b border-border p-5">
                    <div>
                      <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">Document reader</p>
                      <h2 className="mt-1 font-heading text-lg font-semibold">Passages</h2>
                    </div>
                    
                    <div className="flex items-center">
                      <div className={`overflow-hidden transition-all duration-300 ease-out ${isSearchOpen ? 'w-48 opacity-100 mr-2' : 'w-0 opacity-0 mr-0'}`}>
                        <input
                          type="text"
                          placeholder="Search text..."
                          value={searchQuery}
                          onChange={(e) => setSearchQuery(e.target.value)}
                          className="w-full h-10 border border-border bg-background px-3 text-[13px] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
                        />
                      </div>
                      <button 
                        onClick={() => {
                          setIsSearchOpen(!isSearchOpen);
                          if (isSearchOpen) setSearchQuery('');
                        }} 
                        className={`flex min-h-10 min-w-10 items-center justify-center border border-border transition-colors ${isSearchOpen ? 'bg-secondary text-foreground' : 'hover:bg-secondary'}`}
                      >
                        {/* @ts-ignore */}
                        <iconify-icon icon={isSearchOpen ? "ph:x-light" : "ph:magnifying-glass-light"} class="text-lg"></iconify-icon>
                      </button>
                    </div>
                  </div>
                  <div className="p-5 text-base leading-relaxed space-y-4">
                    {result.sentences?.map((sentence, idx) => {
                      if (sentence.band === 'high_ai') {
                        return (
                          <button key={idx} className="group mt-2 w-full border-l-4 border-destructive bg-destructive/10 p-4 text-left leading-relaxed transition-all duration-300 ease-out hover:bg-destructive/20 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring">
                            <span className="font-mono text-[10px] font-semibold uppercase tracking-widest text-destructive transition-transform duration-300 inline-block group-hover:translate-x-0.5">
                              {/* @ts-ignore */}
                              <iconify-icon icon="ph:warning-circle-light" class="mr-1 align-middle text-sm"></iconify-icon>
                              {(sentence.ai_probability * 100).toFixed(0)}% / High AI signal
                            </span>
                            <span className="mt-2 block text-[13px] transition-colors duration-300 group-hover:text-foreground">{renderHighlightedText(sentence.text, searchQuery)}</span>
                          </button>
                        );
                      } else if (sentence.band === 'uncertain') {
                        return (
                          <button key={idx} className="group mt-2 w-full border-l-4 border-[#F5C518] bg-[#F5C518]/20 p-4 text-left leading-relaxed transition-all duration-300 ease-out hover:bg-[#F5C518]/30 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring">
                            <span className="font-mono text-[10px] font-semibold uppercase tracking-widest text-[#78350F] transition-transform duration-300 inline-block group-hover:translate-x-0.5">
                              {/* @ts-ignore */}
                              <iconify-icon icon="ph:warning-circle-light" class="mr-1 align-middle text-sm"></iconify-icon>
                              {(sentence.ai_probability * 100).toFixed(0)}% / Review signal
                            </span>
                            <span className="mt-2 block text-[13px] transition-colors duration-300 group-hover:text-foreground">{renderHighlightedText(sentence.text, searchQuery)}</span>
                          </button>
                        );
                      } else {
                        return (
                          <span key={idx} className="inline mr-1 bg-tertiary/10 rounded-sm px-1 leading-relaxed">{renderHighlightedText(sentence.text, searchQuery)}</span>
                        );
                      }
                    })}
                  </div>
                </article>

                {/* Sidebar Metrics */}
                <aside className="flex flex-col gap-4 lg:col-span-5">
                  <section className="border border-border bg-card p-5">
                    <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">Overall signal</p>
                    <div className="mt-5 flex items-center gap-5">
                      <div className={`flex h-28 w-28 shrink-0 items-center justify-center rounded-full border-8 transition-all duration-500 ease-out hover:scale-[1.02] hover:shadow-md ${result.band === 'high_ai' ? 'border-destructive hover:shadow-destructive/20' : result.band === 'human' ? 'border-tertiary hover:shadow-tertiary/20' : 'border-accent hover:shadow-accent/20'}`}>
                        <div className="text-center">
                          <p className="font-mono text-3xl font-bold">{(result.authenticity_score * 100).toFixed(0)}%</p>
                          <p className="text-[10px] text-muted-foreground">Authenticity</p>
                        </div>
                      </div>
                      <div>
                        <p className="font-heading text-lg font-semibold">{result.band_label || (result.is_ai ? 'High AI signal' : 'Human-led')}</p>
                        <p className="mt-2 text-[13px] leading-5 text-muted-foreground">
                          {result.band === 'high_ai' ? 'Review flagged passages alongside authorship context.' : 'The document largely exhibits human writing patterns.'}
                        </p>
                      </div>
                    </div>
                  </section>
                  
                  <section className="border border-border bg-card">
                    <div className="border-b border-border p-5">
                      <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">Linguistic breakdown</p>
                      <h2 className="mt-1 font-heading text-lg font-semibold">Writing signals</h2>
                    </div>
                    <div className="divide-y divide-border">
                      <div className="flex min-h-14 w-full items-center justify-between px-5 text-left">
                        <span><span className="font-semibold text-[13px]">Human Sentences</span><span className="ml-2 font-mono text-[10px] text-muted-foreground">{result.sentence_distribution.human_count}</span></span>
                      </div>
                      <div className="flex min-h-14 w-full items-center justify-between px-5 text-left">
                        <span><span className="font-semibold text-[13px]">Review Sentences</span><span className="ml-2 font-mono text-[10px] text-muted-foreground">{result.sentence_distribution.uncertain_count}</span></span>
                      </div>
                      <div className="flex min-h-14 w-full items-center justify-between px-5 text-left">
                        <span><span className="font-semibold text-[13px]">AI Sentences</span><span className="ml-2 font-mono text-[10px] text-muted-foreground">{result.sentence_distribution.ai_count}</span></span>
                      </div>
                    </div>
                  </section>
                  
                  {result.feature_contributions && result.feature_contributions.length > 0 && (
                    <section className="border border-border bg-secondary p-5">
                      <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">Feature inspector</p>
                      <h2 className="mt-2 font-heading text-lg font-semibold">Top Indicator</h2>
                      <p className="mt-2 text-[13px] leading-5 text-muted-foreground">
                        {result.feature_contributions[0].feature} had a significant contribution to the final score.
                      </p>
                    </section>
                  )}
                </aside>
                
              </div>
            </section>
            
            {/* Mobile Fixed Bottom Bar */}
            <section className="fixed bottom-0 left-0 right-0 border-t border-border bg-card p-5 shadow-md lg:hidden z-10">
              <div className="mx-auto h-1 w-12 bg-input rounded-full mb-2"></div>
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">Authenticity</p>
                  <h2 className="mt-1 font-heading text-lg font-semibold">{(result.authenticity_score * 100).toFixed(0)}% Match</h2>
                </div>
                <button onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })} className="min-h-11 border border-primary bg-primary px-3 text-[13px] font-semibold text-primary-foreground">
                  View Passages
                </button>
              </div>
            </section>
            </>
          )}

        </main>
      </div>

      {/* How it Works Modal */}
      <div 
        className={`fixed inset-0 z-50 flex items-center justify-center transition-all duration-500 ease-out ${isHowItWorksOpen ? 'opacity-100 pointer-events-auto backdrop-blur-md bg-background/40' : 'opacity-0 pointer-events-none backdrop-blur-none bg-background/0'}`}
        onClick={() => setIsHowItWorksOpen(false)}
      >
        <div 
          className={`relative w-full max-w-lg border border-border bg-card p-6 shadow-2xl transition-all duration-500 delay-100 ease-out ${isHowItWorksOpen ? 'translate-y-0 opacity-100' : 'translate-y-8 opacity-0'}`}
          onClick={(e) => e.stopPropagation()}
        >
          <button onClick={() => setIsHowItWorksOpen(false)} className="absolute right-4 top-4 text-muted-foreground hover:text-foreground transition-colors">
            {/* @ts-ignore */}
            <iconify-icon icon="ph:x-light" class="text-xl"></iconify-icon>
          </button>
          
          <p className="font-mono text-[10px] uppercase tracking-widest text-primary">Instructions</p>
          <h2 className="mt-2 font-heading text-xl font-semibold">How to get the best results</h2>
          
          <div className="mt-6 space-y-5 text-[13px] text-muted-foreground">
            <div className="flex gap-4">
              <span className="flex h-7 w-7 shrink-0 items-center justify-center border border-border bg-secondary font-mono text-[10px] font-bold text-foreground">1</span>
              <p><strong className="text-foreground">Length matters.</strong> AI detection works best on texts that are at least 150-250 words long. Short snippets lack the necessary linguistic variation to map patterns accurately.</p>
            </div>
            <div className="flex gap-4">
              <span className="flex h-7 w-7 shrink-0 items-center justify-center border border-border bg-secondary font-mono text-[10px] font-bold text-foreground">2</span>
              <p><strong className="text-foreground">Avoid extreme formatting.</strong> Paste standard prose and paragraphs. Heavy markdown, bulleted lists, or raw code blocks might skew the mathematical analysis.</p>
            </div>
            <div className="flex gap-4">
              <span className="flex h-7 w-7 shrink-0 items-center justify-center border border-border bg-secondary font-mono text-[10px] font-bold text-foreground">3</span>
              <p><strong className="text-foreground">Use as a signal, not absolute proof.</strong> Our engine highlights passages that exhibit mathematical probability mirroring AI training data. Treat it as a flag for editorial review, not as definitive proof of plagiarism.</p>
            </div>
          </div>
          
          <button onClick={() => setIsHowItWorksOpen(false)} className="mt-8 w-full min-h-11 border border-primary bg-primary px-3 text-[13px] font-semibold text-primary-foreground transition-all hover:bg-foreground">
            Got it
          </button>
        </div>
      </div>
    </div>
  );
};
