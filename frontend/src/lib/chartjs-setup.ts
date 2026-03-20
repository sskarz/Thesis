import {
  Chart,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  ArcElement,
  Tooltip,
  Legend,
  type Plugin,
} from 'chart.js';

Chart.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  ArcElement,
  Tooltip,
  Legend,
);

// Global defaults matching the static dashboard
Chart.defaults.color = '#8a8a8a';
Chart.defaults.borderColor = '#e5e5e3';
Chart.defaults.font.family = "'Source Sans 3', sans-serif";
Chart.defaults.animation = false as any;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyle = 'rectRounded';
Chart.defaults.plugins.legend.labels.padding = 20;
Chart.defaults.plugins.legend.labels.font = { size: 12, weight: 500 as any };
Chart.defaults.plugins.tooltip.backgroundColor = '#1a1a1a';
Chart.defaults.plugins.tooltip.titleFont = { family: "'Source Sans 3', sans-serif", size: 13 };
Chart.defaults.plugins.tooltip.bodyFont = { family: "'JetBrains Mono', monospace", size: 12 };
Chart.defaults.plugins.tooltip.cornerRadius = 6;
Chart.defaults.plugins.tooltip.padding = 10;

// Bar labels plugin — renders values above bars
export const barLabelsPlugin: Plugin<'bar'> = {
  id: 'barLabels',
  afterDatasetsDraw(chart) {
    const fmt = (chart.config.options as any)?._barLabelFmt;
    if (!fmt) return;
    const { ctx } = chart;
    chart.data.datasets.forEach((_ds, i) => {
      const meta = chart.getDatasetMeta(i);
      if (meta.hidden) return;
      meta.data.forEach((el, idx) => {
        const v = chart.data.datasets[i].data[idx];
        if (v == null || typeof v !== 'number') return;
        ctx.save();
        ctx.fillStyle = '#525252';
        ctx.font = "500 11px 'JetBrains Mono', monospace";
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText(fmt(v), el.x, el.y - 6);
        ctx.restore();
      });
    });
  },
};

Chart.register(barLabelsPlugin);
