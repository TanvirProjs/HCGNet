// Helpers reproducing the IEEE conference template (A4) formatting with docx-js.
const fs = require("fs");
const d = require("docx");
const { Paragraph, TextRun, AlignmentType, TabStopType, Table, TableRow, TableCell, WidthType, BorderStyle,
  ImageRun, VerticalAlign, LineRuleType } = d;

const PT = 20; // twips per point
const FONT = "Times New Roman";
const COL_W = 243.3; // pt, width of one body column (A4, 45.35 pt margins, 18 pt gutter)

// inline markup: {i:..} italic, {b:..} bold, {bi:..}, {sub:..}, {sup:..}, {isub:..} italic subscript, {sc:..} small caps
function runs(text, base = {}) {
  const out = [];
  const re = /\{(i|b|bi|sub|sup|isub|isup|sc):([^{}]*)\}/g;
  let last = 0, m;
  const push = (t, o) => { if (t.length) out.push(new TextRun({ text: t, font: FONT, ...base, ...o })); };
  while ((m = re.exec(text)) !== null) {
    push(text.slice(last, m.index), {});
    const k = m[1], t = m[2];
    const o = { i: { italics: true }, b: { bold: true }, bi: { bold: true, italics: true }, sub: { subScript: true },
      sup: { superScript: true }, isub: { subScript: true, italics: true }, isup: { superScript: true, italics: true },
      sc: { smallCaps: true } }[k];
    push(t, o);
    last = re.lastIndex;
  }
  push(text.slice(last), {});
  return out;
}

const body = (text) => new Paragraph({
  children: runs(text, { size: 20, characterSpacing: -1 }),
  alignment: AlignmentType.JUSTIFIED,
  indent: { firstLine: 14.4 * PT },
  spacing: { after: 6 * PT, line: 228, lineRule: LineRuleType.AUTO },
});

const bodyNoIndent = (text) => new Paragraph({
  children: runs(text, { size: 20, characterSpacing: -1 }),
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 6 * PT, line: 228, lineRule: LineRuleType.AUTO },
});

const ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"];
const h1 = (n, text) => new Paragraph({
  children: [new TextRun({ text: (n ? ROMAN[n] + ". " : "") + text, font: FONT, size: 20, smallCaps: true })],
  alignment: AlignmentType.CENTER, keepNext: true, keepLines: true,
  spacing: { before: 8 * PT, after: 4 * PT },
});
const h2 = (letter, text) => new Paragraph({
  children: [new TextRun({ text: `${letter}.  ${text}`, font: FONT, size: 20, italics: true })],
  alignment: AlignmentType.LEFT, keepNext: true, keepLines: true,
  spacing: { before: 6 * PT, after: 3 * PT },
});

// numbered display equation: centre tab + right tab for the number
const eq = (text, n) => new Paragraph({
  children: [new TextRun({ text: "\t", font: FONT, size: 20 }), ...runs(text, { size: 20 }),
    new TextRun({ text: `\t(${n})`, font: FONT, size: 20 })],
  tabStops: [{ type: TabStopType.CENTER, position: Math.round(COL_W / 2 * PT) },
    { type: TabStopType.RIGHT, position: Math.round(COL_W * PT) }],
  spacing: { before: 12 * PT, after: 12 * PT, line: 216, lineRule: LineRuleType.AUTO },
  keepLines: true,
});

const bullet = (text) => new Paragraph({
  children: [new TextRun({ text: "•\t", font: FONT, size: 20 }), ...runs(text, { size: 20, characterSpacing: -1 })],
  alignment: AlignmentType.JUSTIFIED,
  tabStops: [{ type: TabStopType.LEFT, position: Math.round(0.2 * 1440) }],
  indent: { left: Math.round(0.2 * 1440), hanging: Math.round(0.2 * 1440) },
  spacing: { after: 3 * PT, line: 228, lineRule: LineRuleType.AUTO },
});

// table caption: "TABLE I" line and title line, 8 pt small caps, centred
const tableHead = (num, title) => [
  new Paragraph({ children: [new TextRun({ text: `TABLE ${num}`, font: FONT, size: 16, smallCaps: true })],
    alignment: AlignmentType.CENTER, spacing: { before: 8 * PT, after: 0 }, keepNext: true }),
  new Paragraph({ children: runs(title, { size: 16, smallCaps: true }),
    alignment: AlignmentType.CENTER, spacing: { after: 4 * PT }, keepNext: true }),
];

const B1 = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
// rows: array of arrays of strings; first row = header (bold 8 pt). widths in pt.
function table(rows, widths, opts = {}) {
  rows = rows;
  const total = widths.reduce((a, b) => a + b, 0);
  const tw = widths.map(w => Math.round(w * PT));
  const mk = (txt, r, c) => {
    const head = r < (opts.headRows || 1);
    const bold = head || (opts.boldCells || []).some(([rr, cc]) => rr === r && cc === c);
    const align = c === 0 && !opts.centerFirst ? AlignmentType.LEFT : AlignmentType.CENTER;
    const span = (opts.spans || {})[`${r},${c}`] || 1;
    const wsum = tw.slice(c, c + span).reduce((a, b) => a + b, 0);
    return new TableCell({
      width: { size: wsum, type: WidthType.DXA },
      margins: { top: 15, bottom: 15, left: 50, right: 50 },
      verticalAlign: VerticalAlign.CENTER,
      columnSpan: (opts.spans || {})[`${r},${c}`],
      children: [new Paragraph({ children: runs(txt, { size: 16, bold: bold && !String(txt).startsWith("{") ? true : undefined }),
        alignment: align, keepNext: r < rows.length - 1, keepLines: true, spacing: { before: 0, after: 0, line: 200, lineRule: LineRuleType.AUTO } })],
    });
  };
  return new Table({
    width: { size: Math.round(total * PT), type: WidthType.DXA },
    columnWidths: tw,
    alignment: AlignmentType.CENTER,
    borders: { top: B1, bottom: B1, left: B1, right: B1, insideHorizontal: B1, insideVertical: B1 },
    rows: rows.map((row, r) => new TableRow({ cantSplit: true, children: row.map((t, c) => mk(t, r, c)).filter((_, c) => !(opts.skip || {})[`${r},${c}`]) })),
  });
}

const figure = (path, wPt, hPt) => new Paragraph({
  children: [new ImageRun({ type: "png", data: fs.readFileSync(path), transformation: { width: wPt * 96 / 72, height: hPt * 96 / 72 } })],
  alignment: AlignmentType.CENTER, spacing: { before: 4 * PT, after: 0 }, keepNext: true,
});
const figCaption = (n, text) => new Paragraph({
  children: [new TextRun({ text: `Fig. ${n}.  `, font: FONT, size: 16 }), ...runs(text, { size: 16 })],
  alignment: AlignmentType.JUSTIFIED, spacing: { before: 4 * PT, after: 10 * PT },
});

const ref = (n, text) => new Paragraph({
  children: [new TextRun({ text: `[${n}]\t`, font: FONT, size: 16 }), ...runs(text, { size: 16 })],
  alignment: AlignmentType.JUSTIFIED,
  tabStops: [{ type: TabStopType.LEFT, position: Math.round(0.25 * 1440) }],
  indent: { left: Math.round(0.25 * 1440), hanging: Math.round(0.25 * 1440) },
  spacing: { after: Math.round(2.5 * PT), line: 9 * PT, lineRule: LineRuleType.EXACT },
});

// algorithm box: one bordered cell, 8 pt lines with hanging indent for wrapped lines
function algoBox(title, lines) {
  const ps = [new Paragraph({ children: runs(title, { size: 16 }), spacing: { before: 0, after: 20 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "000000", space: 1 } } })];
  lines.forEach((l, i) => ps.push(new Paragraph({
    children: [new TextRun({ text: `${i + 1}:\t`, font: FONT, size: 16 }), ...runs(l, { size: 16 })],
    tabStops: [{ type: TabStopType.LEFT, position: 220 }], indent: { left: 220, hanging: 220 },
    alignment: AlignmentType.LEFT, spacing: { before: 0, after: 0, line: 200, lineRule: LineRuleType.AUTO }, keepNext: i < lines.length - 1, keepLines: true })));
  const w = Math.round(COL_W * PT);
  return new Table({ width: { size: w, type: WidthType.DXA }, columnWidths: [w],
    borders: { top: { style: BorderStyle.SINGLE, size: 8, color: "000000" }, bottom: { style: BorderStyle.SINGLE, size: 8, color: "000000" },
      left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }, right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
      insideHorizontal: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }, insideVertical: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" } },
    rows: [new TableRow({ cantSplit: true, children: [new TableCell({ width: { size: w, type: WidthType.DXA }, margins: { top: 30, bottom: 30, left: 40, right: 40 }, children: ps })] })] });
}

module.exports = { algoBox, d, PT, FONT, COL_W, runs, body, bodyNoIndent, h1, h2, eq, bullet, tableHead, table, figure, figCaption, ref };
