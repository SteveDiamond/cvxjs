import XLSX from 'xlsx';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Asset configuration mapping Simba columns to our asset IDs
const ASSET_CONFIG = [
  { id: 'us_total', name: 'US Total Market', column: 'TSM (US)', ticker: 'VTSAX', color: '#38bdf8' },
  { id: 'us_scv', name: 'US Small Cap Value', column: 'SCV', ticker: 'VSIAX', color: '#fb7185' },
  { id: 'us_lcg', name: 'US Large Cap Growth', column: 'LCG', ticker: 'VIGAX', color: '#a78bfa' },
  { id: 'intl_dev', name: 'Int\'l Developed', column: 'Int\'l Dev', ticker: 'VTMGX', color: '#34d399' },
  { id: 'em', name: 'Emerging Markets', column: 'Emerging', ticker: 'VEMAX', color: '#fbbf24' },
  { id: 'us_bond', name: 'US Total Bond', column: 'TBM (US)', ticker: 'VBTLX', color: '#60a5fa' },
  { id: 'lt_treasury', name: 'Long-Term Treasury', column: 'LTT', ticker: 'VLGSX', color: '#2dd4bf' },
  { id: 'tips', name: 'TIPS', column: 'TIPS', ticker: 'VAIPX', color: '#f472b6' },
  { id: 'reits', name: 'REITs', column: 'REIT', ticker: 'VGSLX', color: '#a3e635' },
  { id: 'gold', name: 'Gold', column: 'Gold', ticker: 'IAU', color: '#facc15' },
];

const START_YEAR = 1974;
const END_YEAR = 2024;
const T_BILL_COLUMN = 'T-Bill';

interface AssetData {
  id: string;
  name: string;
  ticker: string;
  return: number;
  volatility: number;
  color: string;
}

interface SimbaData {
  period: { start: number; end: number };
  riskFreeRate: number;
  assets: AssetData[];
  correlations: number[][];
  annualReturns: Record<string, Record<string, number>>;
}

function parseExcelData(filePath: string): Map<string, Map<number, number>> {
  const workbook = XLSX.readFile(filePath);
  const sheet = workbook.Sheets['Data_Series'];

  if (!sheet) {
    throw new Error('Data_Series sheet not found');
  }

  // Convert to JSON with header
  const data = XLSX.utils.sheet_to_json<Record<string, unknown>>(sheet, { header: 1 });

  // First row has headers (column names)
  const headers = data[0] as string[];

  // Find column indices for each asset we care about
  const columnIndices: Record<string, number> = {};

  // Build mapping of column name to index
  for (let i = 0; i < headers.length; i++) {
    const header = headers[i];
    if (header) {
      columnIndices[header.trim()] = i;
    }
  }

  console.log('Found columns:', Object.keys(columnIndices).slice(0, 20));

  // Extract returns data by asset
  const returnsByAsset = new Map<string, Map<number, number>>();

  // Initialize maps for each asset we need
  const columnsToExtract = [
    ...ASSET_CONFIG.map(a => a.column),
    T_BILL_COLUMN
  ];

  for (const col of columnsToExtract) {
    returnsByAsset.set(col, new Map());
  }

  // Parse data rows (skip header rows - data starts around row 7)
  for (let i = 6; i < data.length; i++) {
    const row = data[i] as (string | number)[];
    if (!row || row.length === 0) continue;

    // First column should be year
    const yearValue = row[0];
    const year = typeof yearValue === 'number' ? yearValue : parseInt(String(yearValue));

    if (isNaN(year) || year < 1800 || year > 2100) continue;
    if (year < START_YEAR || year > END_YEAR) continue;

    // Extract returns for each column we need
    for (const colName of columnsToExtract) {
      const colIndex = columnIndices[colName];
      if (colIndex === undefined) {
        console.warn(`Column not found: ${colName}`);
        continue;
      }

      const value = row[colIndex];
      if (value !== undefined && value !== null && value !== '') {
        // Returns are in percentage form (e.g., 15.30 means 15.30%)
        const returnValue = typeof value === 'number' ? value / 100 : parseFloat(String(value)) / 100;
        if (!isNaN(returnValue)) {
          returnsByAsset.get(colName)!.set(year, returnValue);
        }
      }
    }
  }

  return returnsByAsset;
}

function geometricMean(returns: number[]): number {
  // Calculate geometric mean: (∏(1 + r_i))^(1/n) - 1
  const product = returns.reduce((acc, r) => acc * (1 + r), 1);
  return Math.pow(product, 1 / returns.length) - 1;
}

function standardDeviation(values: number[]): number {
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const squaredDiffs = values.map(v => Math.pow(v - mean, 2));
  const variance = squaredDiffs.reduce((a, b) => a + b, 0) / (values.length - 1);
  return Math.sqrt(variance);
}

function correlationFromYearlyData(
  asset1Returns: Map<number, number>,
  asset2Returns: Map<number, number>,
  years: number[]
): number {
  // Only use years where both assets have data
  const pairs: [number, number][] = [];
  for (const year of years) {
    const r1 = asset1Returns.get(year);
    const r2 = asset2Returns.get(year);
    if (r1 !== undefined && r2 !== undefined) {
      pairs.push([r1, r2]);
    }
  }

  if (pairs.length < 10) return 0;

  const x = pairs.map(p => p[0]);
  const y = pairs.map(p => p[1]);
  const n = pairs.length;

  const meanX = x.reduce((a, b) => a + b, 0) / n;
  const meanY = y.reduce((a, b) => a + b, 0) / n;

  let numerator = 0;
  let sumSqX = 0;
  let sumSqY = 0;

  for (let i = 0; i < n; i++) {
    const dx = x[i] - meanX;
    const dy = y[i] - meanY;
    numerator += dx * dy;
    sumSqX += dx * dx;
    sumSqY += dy * dy;
  }

  const denominator = Math.sqrt(sumSqX * sumSqY);
  return denominator === 0 ? 0 : numerator / denominator;
}

function main() {
  const excelPath = path.join(__dirname, '..', 'Backtest-Portfolio-returns-rev24c.xlsx');
  const outputPath = path.join(__dirname, '..', 'examples', 'demo', 'public', 'data', 'simba-data.json');

  console.log('Reading Excel file:', excelPath);
  const returnsByAsset = parseExcelData(excelPath);

  // Calculate statistics for each asset
  const assets: AssetData[] = [];
  const assetReturnsMaps: Map<number, number>[] = [];
  const annualReturns: Record<string, Record<string, number>> = {};

  // Get years that have data for most assets
  const years: number[] = [];
  for (let y = START_YEAR; y <= END_YEAR; y++) {
    years.push(y);
    annualReturns[y.toString()] = {};
  }

  for (const config of ASSET_CONFIG) {
    const returnsMap = returnsByAsset.get(config.column);
    if (!returnsMap) {
      console.warn(`No data found for asset: ${config.column}`);
      continue;
    }

    // Get returns for the period
    const returnsArray: number[] = [];
    for (const year of years) {
      const r = returnsMap.get(year);
      if (r !== undefined) {
        returnsArray.push(r);
        annualReturns[year.toString()][config.id] = Math.round(r * 10000) / 10000; // Round to 4 decimal places
      }
    }

    if (returnsArray.length < 10) {
      console.warn(`Insufficient data for ${config.column}: only ${returnsArray.length} years`);
      continue;
    }

    const geoMean = geometricMean(returnsArray);
    const vol = standardDeviation(returnsArray);

    console.log(`${config.name}: return=${(geoMean * 100).toFixed(2)}%, vol=${(vol * 100).toFixed(2)}%, years=${returnsArray.length}`);

    assets.push({
      id: config.id,
      name: config.name,
      ticker: config.ticker,
      return: Math.round(geoMean * 10000) / 10000,
      volatility: Math.round(vol * 10000) / 10000,
      color: config.color,
    });

    assetReturnsMaps.push(returnsMap);
  }

  // Calculate correlation matrix using overlapping years
  const n = assets.length;
  const correlations: number[][] = [];

  for (let i = 0; i < n; i++) {
    correlations[i] = [];
    for (let j = 0; j < n; j++) {
      const corr = correlationFromYearlyData(assetReturnsMaps[i], assetReturnsMaps[j], years);
      correlations[i][j] = Math.round(corr * 100) / 100; // Round to 2 decimal places
    }
  }

  console.log('\nCorrelation Matrix:');
  console.log('Assets:', assets.map(a => a.id).join(', '));
  for (let i = 0; i < n; i++) {
    console.log(`${assets[i].id.padEnd(12)}: ${correlations[i].map(c => c.toFixed(2).padStart(6)).join(' ')}`);
  }

  // Calculate risk-free rate from T-Bills (use arithmetic mean for Sharpe ratio)
  const tBillReturns = returnsByAsset.get(T_BILL_COLUMN);
  let riskFreeRate = 0.04; // Default
  if (tBillReturns) {
    const tBillArray: number[] = [];
    for (const year of years) {
      const r = tBillReturns.get(year);
      if (r !== undefined) tBillArray.push(r);
    }
    if (tBillArray.length > 0) {
      const avgTbill = tBillArray.reduce((a, b) => a + b, 0) / tBillArray.length;
      console.log(`\nT-Bill historical average: ${(avgTbill * 100).toFixed(2)}%`);
      // Use 4% as a reasonable long-term assumption for Sharpe ratio
      riskFreeRate = 0.04;
      console.log(`Using risk-free rate: ${(riskFreeRate * 100).toFixed(2)}%`);
    }
  }

  // Build output
  const output: SimbaData = {
    period: { start: START_YEAR, end: END_YEAR },
    riskFreeRate: Math.round(riskFreeRate * 10000) / 10000,
    assets,
    correlations,
    annualReturns,
  };

  // Write to file
  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
  console.log(`\nData written to: ${outputPath}`);
}

main();
