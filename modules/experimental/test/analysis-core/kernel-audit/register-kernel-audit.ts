// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import type {KernelAuditCase} from './kernel-audit-context';
import {auditKernelCase, formatKernelAuditFindings} from './kernel-audit-runner';

/** Registers only one area's builds, so Vitest can schedule audit work across spec files. */
export function registerKernelAudit(area: string, cases: readonly KernelAuditCase[]): void {
  describe(`kernel audit: ${area}`, () => {
    for (const auditCase of cases) {
      it(`${auditCase.exportName} ${auditCase.name}`, () => {
        const result = auditKernelCase(auditCase);
        expect(result.findings, formatKernelAuditFindings(result.findings)).toEqual([]);
        if (!result.rejected) {
          expect(result.nodeCount, 'declared command nodes').toBeGreaterThan(0);
        }
      });
    }
  });
}
