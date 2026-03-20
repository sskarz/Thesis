import { BrowserRouter, Routes, Route } from 'react-router';
import { Layout } from './components/Layout';
import { ComparisonDashboard } from './pages/ComparisonDashboard';
import { ModelDetail } from './pages/ModelDetail';
import { RunsList } from './pages/RunsList';
import { RunDetail } from './pages/RunDetail';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<ComparisonDashboard />} />
          <Route path="model/:modelKey" element={<ModelDetail />} />
          <Route path="runs" element={<RunsList />} />
          <Route path="runs/:runId" element={<RunDetail />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
