import { useEffect, useRef } from "react";
import {
  init,
  use,
  type EChartsCoreOption,
  type EChartsType,
} from "echarts/core";
import { LineChart, ScatterChart } from "echarts/charts";
import {
  AriaComponent,
  AxisPointerComponent,
  GridComponent,
  LegendComponent,
  TitleComponent,
  TooltipComponent,
} from "echarts/components";
import { LabelLayout } from "echarts/features";
import { SVGRenderer } from "echarts/renderers";
import type { LoadTestRequestResult, LoadTestResponse } from "./types";

use([
  LineChart,
  ScatterChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  TitleComponent,
  AxisPointerComponent,
  AriaComponent,
  LabelLayout,
  SVGRenderer,
]);

type Theme = "light" | "dark";

interface LoadTestChartsProps {
  result: LoadTestResponse;
  displayName: string;
  theme: Theme;
  live?: boolean;
}

interface ChartTokens {
  text: string;
  secondaryText: string;
  mutedText: string;
  border: string;
  grid: string;
  surface: string;
  actual: string;
  reference: string;
  ok: string;
  retried: string;
  failed: string;
  fontFamily: string;
}

interface RetryPoint {
  request: LoadTestRequestResult;
  value: [number, number];
}

interface TooltipItem {
  value?: unknown;
  marker?: unknown;
  seriesName?: unknown;
  dataIndex?: unknown;
}

const FAILED_SYMBOL =
  "path://M0,1.35L3.65,5L0,8.65L1.35,10L5,6.35L8.65,10L10,8.65L6.35,5L10,1.35L8.65,0L5,3.65L1.35,0Z";

function formatInteger(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return new Intl.NumberFormat("en-US").format(Math.round(value));
}

function formatSeconds(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return value >= 100 ? `${Math.round(value)}s` : `${value.toFixed(1)}s`;
}

function escapeHtml(value: string): string {
  return value.replace(/[&<>'"]/g, (character) => {
    const entities: Record<string, string> = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      "'": "&#39;",
      '"': "&quot;",
    };
    return entities[character] ?? character;
  });
}

function readToken(style: CSSStyleDeclaration, name: string): string {
  return style.getPropertyValue(name).trim();
}

function getChartTokens(): ChartTokens {
  const rootStyle = getComputedStyle(document.documentElement);
  const bodyStyle = getComputedStyle(document.body);
  return {
    text: readToken(rootStyle, "--text-primary"),
    secondaryText: readToken(rootStyle, "--text-secondary"),
    mutedText: readToken(rootStyle, "--text-muted"),
    border: readToken(rootStyle, "--border-light"),
    grid: readToken(rootStyle, "--border-divider"),
    surface: readToken(rootStyle, "--bg-surface"),
    actual: readToken(rootStyle, "--load-actual"),
    reference: readToken(rootStyle, "--load-reference"),
    ok: readToken(rootStyle, "--load-ok"),
    retried: readToken(rootStyle, "--load-retried"),
    failed: readToken(rootStyle, "--load-failed"),
    fontFamily: bodyStyle.fontFamily,
  };
}

function getMaxElapsed(result: LoadTestResponse): number {
  return Math.max(
    result.summary.wall_time_s,
    result.series[result.series.length - 1]?.elapsed_s ?? 0,
    ...result.requests.map((request) => request.abs_end_s),
    1
  );
}

function tooltipTuple(value: unknown): [number, number] | null {
  if (
    Array.isArray(value) &&
    value.length >= 2 &&
    typeof value[0] === "number" &&
    typeof value[1] === "number"
  ) {
    return [value[0], value[1]];
  }
  return null;
}

function tooltipItems(value: unknown): TooltipItem[] {
  const values = Array.isArray(value) ? value : [value];
  return values.filter(
    (item): item is TooltipItem => typeof item === "object" && item !== null
  );
}

function formatOutputTooltip(params: unknown): string {
  const items = tooltipItems(params);
  const firstValue = tooltipTuple(items[0]?.value);
  const heading = firstValue ? formatSeconds(firstValue[0]) : "Observed run";
  const rows = items
    .map((item) => {
      const value = tooltipTuple(item.value);
      if (!value) return "";
      const marker = typeof item.marker === "string" ? item.marker : "";
      const seriesName =
        typeof item.seriesName === "string" ? item.seriesName : "Value";
      return `${marker}${escapeHtml(seriesName)}: <strong>${formatInteger(value[1])}</strong>`;
    })
    .filter(Boolean);
  return `<strong>${heading}</strong><br/>${rows.join("<br/>")}`;
}

function formatRequestTooltip(point: RetryPoint): string {
  const { request } = point;
  const retries = Math.max(request.attempts - 1, 0);
  const ok = request.status.toLowerCase() === "ok";
  const outcome = ok
    ? retries > 0
      ? "Succeeded after retry"
      : "Succeeded on first try"
    : "Failed";
  const latestRetry = request.retry_events?.[request.retry_events.length - 1];
  const retryReason = latestRetry
    ? latestRetry.status_code
      ? `HTTP ${latestRetry.status_code}`
      : latestRetry.error_code ?? "Invalid response"
    : null;
  const details = [
    `<strong>Request ${request.req_id + 1}</strong>`,
    escapeHtml(outcome),
    `Completed: ${formatSeconds(request.abs_end_s)}`,
    `Latency: ${formatSeconds(request.elapsed_s)}`,
    `Attempts: ${request.attempts}`,
    retryReason ? `Latest retry: ${escapeHtml(retryReason)}` : "",
    !ok && request.error ? `Error: ${escapeHtml(request.error)}` : "",
  ].filter(Boolean);
  return details.join("<br/>");
}

function buildChartOption(
  result: LoadTestResponse,
  displayName: string,
  tokens: ChartTokens,
  reduceMotion: boolean
): EChartsCoreOption {
  const maxElapsed = getMaxElapsed(result);
  const xInterval = maxElapsed / 4;
  const outputReferenceOtpm =
    result.summary.target_otpm ?? result.config.target_otpm ?? null;
  const outputReferenceAtWallClock =
    outputReferenceOtpm == null ? null : (outputReferenceOtpm * maxElapsed) / 60;
  const maxTokens =
    Math.max(
      result.summary.total_completion_tokens,
      outputReferenceAtWallClock ?? 0,
      ...result.series.map((point) => point.cumulative_completion_tokens),
      1
    ) * 1.08;

  const actualData: Array<
    [number, number] | {
      value: [number, number];
      symbolSize: number;
      label: {
        show: boolean;
        position: "top";
        align: "right";
        distance: number;
        formatter: string;
        color: string;
        fontSize: number;
        fontWeight: number;
      };
    }
  > = [
    [0, 0],
    ...result.series.map(
      (point): [number, number] => [
        point.elapsed_s,
        point.cumulative_completion_tokens,
      ]
    ),
  ];
  const finalSeriesPoint = result.series[result.series.length - 1];
  if (finalSeriesPoint) {
    actualData[actualData.length - 1] = {
      value: [
        finalSeriesPoint.elapsed_s,
        finalSeriesPoint.cumulative_completion_tokens,
      ],
      symbolSize: 9,
      label: {
        show: true,
        position: "top",
        align: "right",
        distance: 8,
        formatter: `${formatInteger(finalSeriesPoint.cumulative_completion_tokens)} total`,
        color: tokens.actual,
        fontSize: 12,
        fontWeight: 700,
      },
    };
  }

  const requestPoints: RetryPoint[] = result.requests.map((request) => ({
    request,
    value: [request.abs_end_s, Math.max(request.attempts - 1, 0)],
  }));
  const firstTryPoints = requestPoints.filter(
    ({ request }) =>
      request.status.toLowerCase() === "ok" && request.attempts <= 1
  );
  const retriedPoints = requestPoints.filter(
    ({ request }) =>
      request.status.toLowerCase() === "ok" && request.attempts > 1
  );
  const failedPoints = requestPoints.filter(
    ({ request }) => request.status.toLowerCase() !== "ok"
  );
  const maxRetries = Math.max(
    ...requestPoints.map((point) => point.value[1]),
    0
  );
  const retryInterval = Math.max(1, Math.ceil(maxRetries / 4));
  const retryAxisMax =
    maxRetries === 0
      ? 0.5
      : Math.max(retryInterval, Math.ceil(maxRetries / retryInterval) * retryInterval);
  const retryAxisMin = maxRetries === 0 ? -0.5 : 0;

  const retrySeries = (
    name: string,
    points: RetryPoint[],
    symbol: string,
    color: string,
    symbolSize: number
  ) => ({
    name,
    type: "scatter" as const,
    xAxisIndex: 1,
    yAxisIndex: 1,
    data: points.map((point) => point.value),
    symbol,
    symbolSize,
    itemStyle: {
      color,
      borderColor: tokens.surface,
      borderWidth: symbol === FAILED_SYMBOL ? 0 : 1.5,
    },
    emphasis: {
      scale: 1.35,
      itemStyle: { color },
    },
    tooltip: {
      trigger: "item" as const,
      formatter: (params: unknown) => {
        const item = tooltipItems(params)[0];
        const dataIndex =
          typeof item?.dataIndex === "number" ? item.dataIndex : -1;
        const point = points[dataIndex];
        return point ? formatRequestTooltip(point) : name;
      },
    },
  });

  const outputSeries = [
    {
      name: "Cumulative output tokens",
      type: "line" as const,
      xAxisIndex: 0,
      yAxisIndex: 0,
      data: actualData,
      step: "end" as const,
      showSymbol: true,
      symbol: "circle",
      symbolSize: 4,
      clip: false,
      lineStyle: {
        color: tokens.actual,
        width: 3,
        cap: "round" as const,
        join: "round" as const,
      },
      itemStyle: {
        color: tokens.actual,
        borderColor: tokens.surface,
        borderWidth: 1.25,
      },
      emphasis: { focus: "series" as const },
    },
    ...(outputReferenceAtWallClock == null
      ? []
      : [
          {
            name: `Trajectory at ${formatInteger(outputReferenceOtpm)} tokens/min`,
            type: "line" as const,
            xAxisIndex: 0,
            yAxisIndex: 0,
            data: [
              [0, 0],
              [maxElapsed, outputReferenceAtWallClock],
            ],
            showSymbol: false,
            silent: true,
            lineStyle: {
              color: tokens.reference,
              width: 2.5,
              type: "dashed" as const,
            },
            emphasis: { disabled: true },
          },
        ]),
  ];

  const ariaDescription = `${displayName} produced ${formatInteger(
    result.summary.total_completion_tokens
  )} output tokens over ${formatSeconds(maxElapsed)}. ${
    result.summary.retried
  } requests retried and ${result.summary.failed} failed.`;

  const sharedXAxis = {
    type: "value" as const,
    min: 0,
    max: maxElapsed,
    interval: xInterval,
    boundaryGap: [0, 0],
    axisLine: { show: false },
    axisTick: { show: false },
    splitLine: {
      show: true,
      lineStyle: { color: tokens.grid, type: "dashed" as const, opacity: 0.7 },
    },
    axisPointer: {
      show: true,
      snap: false,
      lineStyle: { color: tokens.mutedText, width: 1, opacity: 0.6 },
      label: {
        backgroundColor: tokens.text,
        color: tokens.surface,
        formatter: ({ value }: { value: number | string }) =>
          formatSeconds(Number(value)),
      },
    },
  };

  return {
    backgroundColor: "transparent",
    animation: !reduceMotion,
    animationDuration: reduceMotion ? 0 : 240,
    animationDurationUpdate: reduceMotion ? 0 : 160,
    textStyle: {
      color: tokens.text,
      fontFamily: tokens.fontFamily,
    },
    aria: {
      enabled: true,
      description: ariaDescription,
    },
    title: [
      {
        text: "Retries over wall-clock time",
        left: 76,
        top: "66%",
        textStyle: {
          color: tokens.text,
          fontFamily: tokens.fontFamily,
          fontSize: 13,
          fontWeight: 650,
        },
      },
    ],
    legend: {
      data: outputSeries.map((series) => series.name),
      left: 76,
      top: 14,
      selectedMode: false,
      itemWidth: 24,
      itemHeight: 8,
      itemGap: 18,
      icon: "roundRect",
      textStyle: {
        color: tokens.secondaryText,
        fontFamily: tokens.fontFamily,
        fontSize: 12,
        fontWeight: 600,
      },
    },
    tooltip: {
      trigger: "axis",
      appendToBody: true,
      confine: true,
      backgroundColor: tokens.surface,
      borderColor: tokens.border,
      borderWidth: 1,
      padding: [10, 12],
      textStyle: {
        color: tokens.text,
        fontFamily: tokens.fontFamily,
        fontSize: 12,
      },
      axisPointer: { type: "line" },
      formatter: formatOutputTooltip,
    },
    axisPointer: {
      link: [{ xAxisIndex: [0, 1] }],
    },
    grid: [
      {
        left: 76,
        right: 28,
        top: 62,
        height: "49%",
        containLabel: false,
      },
      {
        left: 76,
        right: 28,
        top: "73%",
        bottom: 44,
        containLabel: false,
      },
    ],
    xAxis: [
      {
        ...sharedXAxis,
        gridIndex: 0,
        axisLabel: { show: false },
        axisPointer: {
          ...sharedXAxis.axisPointer,
          label: { ...sharedXAxis.axisPointer.label, show: false },
        },
      },
      {
        ...sharedXAxis,
        gridIndex: 1,
        name: "Wall-clock seconds",
        nameLocation: "middle",
        nameGap: 30,
        nameTextStyle: {
          color: tokens.secondaryText,
          fontFamily: tokens.fontFamily,
          fontSize: 11,
          fontWeight: 650,
        },
        axisLabel: {
          show: true,
          color: tokens.mutedText,
          fontSize: 11,
          margin: 10,
          hideOverlap: true,
          formatter: (value: number) =>
            maxElapsed < 10 ? `${value.toFixed(1)}s` : `${Math.round(value)}s`,
        },
      },
    ],
    yAxis: [
      {
        type: "value",
        gridIndex: 0,
        min: 0,
        max: maxTokens,
        splitNumber: 4,
        name: "Output tokens",
        nameLocation: "middle",
        nameGap: 54,
        nameTextStyle: {
          color: tokens.secondaryText,
          fontFamily: tokens.fontFamily,
          fontSize: 11,
          fontWeight: 650,
        },
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: {
          color: tokens.mutedText,
          fontSize: 11,
          margin: 12,
          formatter: (value: number) => formatInteger(value),
        },
        splitLine: {
          show: true,
          lineStyle: { color: tokens.grid, opacity: 0.8 },
        },
      },
      {
        type: "value",
        gridIndex: 1,
        min: retryAxisMin,
        max: retryAxisMax,
        interval: maxRetries === 0 ? 0.5 : retryInterval,
        name: "Retries",
        nameLocation: "middle",
        nameGap: 54,
        nameTextStyle: {
          color: tokens.secondaryText,
          fontFamily: tokens.fontFamily,
          fontSize: 11,
          fontWeight: 650,
        },
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: {
          color: tokens.mutedText,
          fontSize: 11,
          margin: 12,
          formatter: (value: number) =>
            Number.isInteger(value) && value >= 0 ? String(value) : "",
        },
        splitLine: {
          show: true,
          lineStyle: { color: tokens.grid, opacity: 0.8 },
        },
      },
    ],
    series: [
      ...outputSeries,
      retrySeries("First try", firstTryPoints, "circle", tokens.ok, 7),
      retrySeries("Succeeded after retry", retriedPoints, "triangle", tokens.retried, 10),
      retrySeries("Failed", failedPoints, FAILED_SYMBOL, tokens.failed, 11),
    ],
  };
}

export default function LoadTestCharts({
  result,
  displayName,
  theme,
  live = false,
}: LoadTestChartsProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<EChartsType | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const chart = init(container, undefined, { renderer: "svg" });
    chartRef.current = chart;
    const resizeObserver = new ResizeObserver(() => chart.resize());
    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    const frame = requestAnimationFrame(() => {
      const chart = chartRef.current;
      if (!chart) return;
      const reduceMotion = window.matchMedia(
        "(prefers-reduced-motion: reduce)"
      ).matches;
      chart.setOption(
        buildChartOption(
          result,
          displayName,
          getChartTokens(),
          reduceMotion
        ),
        { notMerge: true }
      );
      chart.resize();
    });
    return () => cancelAnimationFrame(frame);
  }, [displayName, result, theme]);

  const summary = result.summary;
  const ariaLabel = `${live ? "Live " : ""}${displayName} load-test charts: ${formatInteger(
    summary.total_completion_tokens
  )} cumulative output tokens, ${summary.retried} retried requests, ${
    summary.failed
  } failed requests.`;

  return (
    <div className="load-test-chart-shell">
      <div className="load-test-chart-header">
        <div>
          <span className="load-test-chart-kicker">{live ? "Live run" : "Observed run"}</span>
          <h3>
            {displayName} — {formatInteger(summary.total_completion_tokens)} cumulative
            output tokens{live ? " so far" : ""}
          </h3>
          <p>
            {live
              ? "Updating from completed requests. Actual output and retries share one wall-clock axis."
              : "Actual output accumulation and request retries share one wall-clock axis."}
            {" "}The dashed line is the configured OTPM reference trajectory.
          </p>
        </div>
        <div className="load-test-chart-meta">
          <strong>
            {live ? `${result.requests.length} of ` : ""}{result.config.total_requests} requests · {result.config.max_workers} max in flight · {result.config.cache_bust ? "cache-bust" : "cache reuse"} · {result.config.scheduling_mode === "target_rate" ? `${result.config.load_multiplier ?? 1}× target rate` : "burst"}
          </strong>
          <span>
            {live ? "Observed throughput so far" : "Average throughput"}: {formatInteger(summary.achieved_otpm)} OTPM over {" "}
            {formatSeconds(summary.wall_time_s)}.
          </span>
        </div>
      </div>

      <div className="load-test-chart-panel composite">
        <div
          ref={containerRef}
          className="load-test-echart"
          role="img"
          aria-label={ariaLabel}
          tabIndex={0}
        />
      </div>
    </div>
  );
}
