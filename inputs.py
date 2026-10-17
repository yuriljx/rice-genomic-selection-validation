from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import platform
import posixpath
import re
import sys
import uuid
import xml.etree.ElementTree as ET
import xml.sax
from xml.sax.handler import ContentHandler, feature_external_ges, feature_external_pes
from zipfile import ZipFile
MISSING_TOKENS = ('', 'na', 'n/a', 'nan', 'null', 'none', '.', '-')
NS_MAIN = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
NS_REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
COLUMN_PATTERN = re.compile('[A-Z]{1,3}')
CELL_PATTERN = re.compile('([A-Z]{1,3})([1-9][0-9]*)')

class InputError(RuntimeError):
    pass

class StopHeaderScan(Exception):
    pass

@dataclass(frozen=True)
class RawCell:
    reference: str
    kind: str
    raw: str
    has_formula: bool = False
    has_cached_value: bool = False

class CellSelector(ContentHandler):

    def __init__(self, rows, columns, *, stop_after_row=None):
        super().__init__()
        self.rows = rows
        self.columns = columns
        self.stop_after_row = stop_after_row
        self.result = {}
        self.row = None
        self.allowed_row = False
        self.cell = None
        self.capture_tag = None
        self.pieces = []
        self.has_formula = False
        self.has_cached_value = False
        self.rph_depth = 0
        self.previous_row = 0
        self.retained_cells = 0

    def startElement(self, name, attrs):
        name = name.rsplit(':', 1)[-1]
        if name == 'row':
            if not re.fullmatch('[1-9][0-9]*', attrs.get('r', '')):
                raise InputError('Worksheet row lacks an explicit valid row index')
            self.row = int(attrs['r'])
            if self.row <= self.previous_row:
                raise InputError('Worksheet rows are duplicated or out of order')
            self.previous_row = self.row
            if self.stop_after_row is not None and self.row > self.stop_after_row:
                raise StopHeaderScan()
            self.allowed_row = self.rows is None or self.row in self.rows
            if self.allowed_row:
                self.result[self.row] = {}
        elif name == 'c':
            self.cell = None
            self.capture_tag = None
            self.pieces = []
            self.has_formula = False
            self.has_cached_value = False
            if not self.allowed_row:
                return
            match = CELL_PATTERN.fullmatch(attrs.get('r', ''))
            if not match or int(match[2]) != self.row:
                raise InputError('Selected row contains an invalid cell coordinate')
            if self.columns is None or match[1] in self.columns:
                self.cell = (attrs['r'], attrs.get('t', 'n'), match[1])
        elif self.cell is not None:
            if name == 'rPh':
                self.rph_depth += 1
            elif name == 'f':
                self.has_formula = True
            elif name == 'v':
                self.has_cached_value = True
                self.capture_tag = 'v'
            elif name == 't' and (not self.rph_depth):
                self.capture_tag = 't'

    def characters(self, content):
        if self.cell is not None and self.capture_tag is not None:
            self.pieces.append(content)

    def endElement(self, name):
        name = name.rsplit(':', 1)[-1]
        if name in ('v', 't'):
            self.capture_tag = None
        elif name == 'rPh' and self.cell is not None:
            self.rph_depth -= 1
        elif name == 'c':
            if self.cell is not None:
                ref, kind, col = self.cell
                if col in self.result[self.row]:
                    raise InputError('Duplicate selected cell coordinate')
                self.result[self.row][col] = RawCell(ref, kind, ''.join(self.pieces), self.has_formula, self.has_cached_value)
                self.retained_cells += 1
            self.cell = None
            self.pieces = []
        elif name == 'row':
            if self.stop_after_row is not None and self.row == self.stop_after_row:
                raise StopHeaderScan()
            self.row = None
            self.allowed_row = False

class SelectedStrings(ContentHandler):

    def __init__(self, wanted):
        super().__init__()
        self.wanted = wanted
        self.index = -1
        self.allowed = False
        self.capture = False
        self.rph_depth = 0
        self.pieces = []
        self.result = {}

    def startElement(self, name, attrs):
        name = name.rsplit(':', 1)[-1]
        if name == 'si':
            self.index += 1
            self.allowed = self.index in self.wanted
            self.pieces = []
        elif name == 'rPh':
            self.rph_depth += 1
        elif name == 't' and (not self.rph_depth):
            self.capture = self.allowed

    def characters(self, content):
        if self.capture:
            self.pieces.append(content)

    def endElement(self, name):
        name = name.rsplit(':', 1)[-1]
        if name == 't':
            self.capture = False
        elif name == 'rPh':
            self.rph_depth -= 1
        elif name == 'si':
            if self.allowed:
                self.result[self.index] = ''.join(self.pieces)
            self.allowed = False
            self.pieces = []

def sax_member(book, member, handler):
    parser = xml.sax.make_parser()
    parser.setFeature(feature_external_ges, False)
    try:
        parser.setFeature(feature_external_pes, False)
    except xml.sax.SAXNotSupportedException:
        pass
    parser.setContentHandler(handler)
    with book.open(member) as stream:
        try:
            parser.parse(stream)
        except StopHeaderScan:
            pass
        except xml.sax.SAXParseException as exc:
            raise InputError('Malformed source XML; selected-source parsing stopped') from None
    return handler

def selected_cells(book, member, rows=None, columns=None, *, stop_after_row=None):
    return sax_member(book, member, CellSelector(rows, columns, stop_after_row=stop_after_row)).result

def resolve_strings(book, raw_rows):
    wanted = set()
    for cells in raw_rows.values():
        for cell in cells.values():
            if cell.kind == 's':
                if not re.fullmatch('[0-9]+', cell.raw):
                    raise InputError('Selected cell has an invalid shared-string reference')
                wanted.add(int(cell.raw))
    if not wanted:
        return {}
    if 'xl/sharedStrings.xml' not in book.namelist():
        raise InputError('Selected cells reference a missing shared-string table')
    result = sax_member(book, 'xl/sharedStrings.xml', SelectedStrings(wanted)).result
    if wanted != set(result):
        raise InputError('Selected shared-string index is absent')
    return result

def cell_text(cell, strings):
    if cell is None:
        return ''
    return strings[int(cell.raw)] if cell.kind == 's' else cell.raw

def parse_yield(cell, strings):
    if cell is None:
        return (None, 'missing_cell')
    if cell.has_formula and (not cell.has_cached_value):
        return (None, 'formula_without_cached_value')
    if cell.kind == 'e':
        return (None, 'excel_error')
    if cell.kind not in ('n', 's', 'str', 'inlineStr'):
        return (None, 'invalid_cell_type')
    value = cell_text(cell, strings).strip()
    if value.casefold() in MISSING_TOKENS:
        return (None, 'missing_token')
    try:
        numeric = float(value)
    except (ValueError, OverflowError):
        return (None, 'nonnumeric')
    if not math.isfinite(numeric):
        return (None, 'nonfinite')
    if numeric < 0:
        return (None, 'negative')
    suffix = '_cached_formula' if cell.has_formula else ''
    if numeric == 0:
        return (numeric, 'zero_flag' + suffix)
    return (numeric, 'valid' + suffix)

def read_wei(path, resource_dir, needed):
    import gzip
    from io import BytesIO
    cfg = json.loads((Path(resource_dir) / 'inputs.json').read_text())
    if hash_file(path)['sha256'] != cfg['xlsx_sha256']:
        raise InputError('Data S4 workbook SHA-256 differs from the documented version')
    with gzip.open(Path(resource_dir) / 'wei_ID_rows.csv.gz', 'rt', newline='') as f:
        registry = {r['normalized_id']: r for r in csv.DictReader(f)}
    ids = {i for i, site in needed}
    if not ids <= registry.keys():
        raise InputError('A required scoring ID is absent from the row map')
    aliases = cfg['explicit_id_aliases']
    with ZipFile(path) as book:
        workbook = ET.fromstring(book.read('xl/workbook.xml'))
        sheets = workbook.find('{' + NS_MAIN + '}sheets')
        targets = [n for n in sheets if n.get('name') == cfg['sheet_name']]
        if len(targets) != 1:
            raise InputError('Data S4 sheet name is not unique')
        relation = targets[0].get('{' + NS_REL + '}id')
        rels = ET.fromstring(book.read('xl/_rels/workbook.xml.rels'))
        target = next(n.get('Target') for n in rels if n.get('Id') == relation)
        member = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)
        byrow = {int(registry[i]['source_row']): i for i in ids}
        if len(byrow) != len(ids):
            raise InputError('Scoring row mapping is not one to one')
        raw = selected_cells(book, member, set(byrow), {'B', 'K', 'Z'})
        strings = resolve_strings(book, raw)
        out = {}
        for row, identifier in byrow.items():
            actual = cell_text(raw.get(row, {}).get('B'), strings).strip()
            for suffix in ('.rlt.bin', '.rlt'):
                if actual.endswith(suffix):
                    actual = actual[:-len(suffix)]
                    break
            actual = aliases.get(actual, actual)
            if actual != identifier:
                raise InputError('Data S4 row-to-ID binding differs')
            for site, column in [('SH', 'K'), ('HZ', 'Z')]:
                if (identifier, site) in needed:
                    value, status = parse_yield(raw.get(row, {}).get(column), strings)
                    if value is None:
                        raise InputError('Required scoring phenotype is undefined: ' + status)
                    out[(identifier, site)] = value
        if set(out) != set(needed):
            raise InputError('Data S4 scoring keys are incomplete')
        return out

def read_irri(path, needed):
    from io import TextIOWrapper
    if hash_file(path)['sha256'] != 'bb138b26474633f89b8f0bd6f8c9495c35a153f157857f7e571425f44bb74cfe':
        raise InputError('IRRI corrected plot ZIP SHA-256 differs')
    records = {}
    with ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith('.csv') and not any(part.startswith(('.', '__')) for part in PurePosixPath(n).parts)]
        if len(names) != 8:
            raise InputError('Expected eight corrected seasonal plot files')
        for name in names:
            with archive.open(name) as raw:
                reader = csv.DictReader(TextIOWrapper(raw, encoding='utf-8-sig', newline=''))
                if not {'GHID', 'YEAR', 'SEASON', 'REP', 'PLOTNO', 'Kg_ha'} <= set(reader.fieldnames or []):
                    raise InputError('Required IRRI plot fields are missing')
                for row in reader:
                    season = row['YEAR'].strip() + {'1':'DS', '2':'WS'}.get(row['SEASON'].strip(), '?')
                    key = (row['GHID'].strip(), season)
                    if key not in needed:
                        continue
                    rep = row['REP'].strip()
                    if not rep:
                        raise InputError('Missing replicate identity')
                    try:
                        value = float(row['Kg_ha'])
                    except (ValueError, OverflowError):
                        continue
                    if not math.isfinite(value) or value < 0:
                        continue
                    pair = records.setdefault(key, {})
                    if rep in pair:
                        raise InputError('Duplicate finite replicate in required IRRI key')
                    pair[rep] = value
    if set(records) != set(needed) or any(len(v) < 2 for v in records.values()):
        raise InputError('Required qualified IRRI means are incomplete')
    return {k:math.fsum(v / len(rr) for v in rr.values()) for k, rr in records.items()}

def hash_file(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 ** 2), b''):
            size += len(block)
            digest.update(block)
    return {'bytes': size, 'sha256': digest.hexdigest()}
