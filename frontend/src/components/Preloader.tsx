import { useEffect, useState } from 'react';

const LOGO_PATTERN = [
  "O.OOO.....BBB.",
  ".YYOO....BBBBB",
  "YY.YO...KK.BB.",
  "YYYYY..KKK.BBB",
  "BYYYY.KKKKKKKB",
  "BB.Y.KKKK.KKKK",
  "BBB..KKK..YKK.",
  "BB...KK...YYKK",
  "B....K....YYYY"
];

const getColorClass = (char: string) => {
  switch (char) {
    case 'O': return 'bg-orange-500';
    case 'Y': return 'bg-yellow-400';
    case 'B': return 'bg-blue-500';
    case 'D': return 'bg-indigo-900';
    case 'K': return 'bg-black';
    default: return 'bg-transparent';
  }
};

interface PreloaderProps {
  onComplete: () => void;
}

export const Preloader = ({ onComplete }: PreloaderProps) => {
  const [animatingOut, setAnimatingOut] = useState(false);

  const [stableDelays] = useState(() => {
    const delays: string[][] = [];
    for (let r = 0; r < 9; r++) {
      delays[r] = [];
      for (let c = 0; c < 14; c++) {
        // Organic radial crystallization (center outwards)
        const dy = r - 4;
        const dx = c - 6.5;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const randomScatter = Math.random() * 400;
        
        delays[r][c] = `${(dist * 100) + randomScatter}ms`;
      }
    }
    return delays;
  });

  const [shatterDelays] = useState(() => {
    const delays: string[][] = [];
    for (let r = 0; r < 9; r++) {
      delays[r] = [];
      for (let c = 0; c < 14; c++) {
        // Radial shockwave delay (center explodes first)
        const dy = r - 4;
        const dx = c - 6.5;
        const dist = Math.sqrt(dx * dx + dy * dy);
        delays[r][c] = `${dist * 40}ms`;
      }
    }
    return delays;
  });

  const [floatVars] = useState(() => {
    const vars: any[][] = [];
    for (let r = 0; r < 9; r++) {
      vars[r] = [];
      for (let c = 0; c < 14; c++) {
        // Float gently like canvas particles (drifting randomly)
        const dx = (Math.random() - 0.5) * 40; // vw
        const dy = (Math.random() - 0.5) * 40; // vh
        
        vars[r][c] = {
          '--dx': `${dx}vw`,
          '--dy': `${dy}vh`,
          '--rot': `0deg` // Canvas pixels don't rotate
        };
      }
    }
    return vars;
  });

  useEffect(() => {

    // List of assets to preload (based on the user's setup)
    const videosToPreload = [
      '/assets/work/vid1.mp4',
      '/assets/work/vid2.mp4',
      '/assets/work/vid3.mp4',
      '/assets/work/vid4.mp4',
      '/assets/work/vid5.mp4',
      '/assets/work/vid7.mp4',
      '/assets/work/vid8.mp4',
      '/assets/work/vid9.mp4',
      '/assets/work/vid10.mp4',
      '/assets/experiments/exp1.mp4',
      '/assets/experiments/exp2.mp4',
      '/assets/experiments/exp3.mp4',
      '/assets/experiments/exp4.mp4',
    ];
    
    // Simulate a minimum loading time for the animation to play out nicely
    const minLoadTime = new Promise(resolve => setTimeout(resolve, 5500));
    
    // Preload videos into cache
    const assetPromises = videosToPreload.map(src => {
      return fetch(src).then(res => res.blob());
    });

    const initSequence = async () => {
      // Wait for min load time AND asset loading
      await Promise.all([minLoadTime, ...assetPromises]);
      
      setAnimatingOut(true);
      setTimeout(() => {
        onComplete();
      }, 1600);
    };

    initSequence();
  }, [onComplete]);

  return (
    <div className={`fixed inset-0 z-[9999] flex flex-col items-center justify-center transition-colors duration-1000 ease-in-out ${animatingOut ? 'bg-transparent pointer-events-none' : 'bg-background'}`}>
      
      <style>{`
        @keyframes assemble {
          0% {
            transform: translate(var(--dx), var(--dy));
            opacity: 0;
            filter: blur(4px);
          }
          100% {
            transform: translate(0px, 0px);
            opacity: 1;
            filter: blur(0);
          }
        }
        @keyframes scatter {
          0% { transform: translate(0px, 0px); opacity: 1; filter: blur(0); }
          100% { transform: translate(var(--dx), var(--dy)); opacity: 0; filter: blur(4px); }
        }
      `}</style>

      <div className="flex flex-col">
        {LOGO_PATTERN.map((row, rIdx) => (
          <div key={rIdx} className="flex">
          {row.split('').map((cell, cIdx) => (
            <div 
              key={`${rIdx}-${cIdx}-outer`}
              style={cell !== '.' ? {
                ...floatVars[rIdx][cIdx],
                animation: `assemble 2400ms cubic-bezier(0.8, 0, 0.2, 1) both`,
                animationDelay: stableDelays[rIdx][cIdx]
              } as React.CSSProperties : { opacity: 0 }}
            >
              <div 
                className={`h-1 w-1 sm:h-[6px] sm:w-[6px] md:h-2 md:w-2 ${cell !== '.' ? `${getColorClass(cell)} shadow-[0_0_10px_rgba(0,0,0,0.15)]` : 'bg-transparent'}`}
                style={cell !== '.' ? {
                  ...floatVars[rIdx][cIdx],
                  animation: animatingOut ? `scatter 1600ms cubic-bezier(0.1, 1.0, 0.3, 1.0) forwards` : 'none',
                  animationDelay: animatingOut ? shatterDelays[rIdx][cIdx] : '0ms'
                } as React.CSSProperties : {}}
              />
            </div>
          ))}
          </div>
        ))}
      </div>
    </div>
  );
};
