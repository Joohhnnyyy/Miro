import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Hero } from './components/Hero';
import { Intro } from './components/Intro';
import { WorkCarousel } from './components/WorkCarousel';
import { Footer } from './components/Footer';
import { Detector } from './components/Detector';
import './index.css';

// Global flag to track if we've navigated to the app portion
declare global {
  interface Window {
    hasNavigatedAwayFromHome: boolean;
  }
}

function Home() {
  import('react').then(({ useEffect }) => {
    useEffect(() => {
      if (window.hasNavigatedAwayFromHome) {
        window.location.reload();
      }
    }, []);
  });

  return (
    <main id="top">
      <Hero />
      <Intro />
      <WorkCarousel />
      <Footer />
    </main>
  );
}

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/detect" element={<Detector />} />
      </Routes>
    </Router>
  );
}

export default App;
