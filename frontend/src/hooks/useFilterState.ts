import { useState, useCallback } from 'react';

export interface FilterState {
  selectedCats: Set<string>;
  selectedModels: Set<string>;
  metric: 'mean' | 'median';
  detailCat: string;
}

export function useFilterState(allCats: string[], allModelKeys: string[]) {
  const [selectedCats, setSelectedCats] = useState<Set<string>>(() => new Set(allCats));
  const [selectedModels, setSelectedModels] = useState<Set<string>>(() => new Set(allModelKeys));
  const [metric, setMetric] = useState<'mean' | 'median'>('mean');
  const [detailCat, setDetailCat] = useState('simple');

  const toggleCat = useCallback((cat: string) => {
    setSelectedCats(prev => {
      const next = new Set(prev);
      next.has(cat) ? next.delete(cat) : next.add(cat);
      return next;
    });
  }, []);

  const toggleModel = useCallback((key: string) => {
    setSelectedModels(prev => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  }, []);

  const selectAllCats = useCallback(() => setSelectedCats(new Set(allCats)), [allCats]);
  const clearCats = useCallback(() => setSelectedCats(new Set()), []);

  return {
    selectedCats, selectedModels, metric, detailCat,
    toggleCat, toggleModel, setMetric, setDetailCat,
    selectAllCats, clearCats,
  };
}
