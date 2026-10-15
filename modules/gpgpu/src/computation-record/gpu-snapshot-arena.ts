// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type CommandEncoder, type Device} from '@luma.gl/core';
import type {
  GPUComputationCapture,
  GPUComputationCaptureElementType
} from './computation-record-types';

/** One packed, four-byte-aligned range in every snapshot frame. */
export type GPUSnapshotSlot = {
  /** Unique name within the arena, also used as the capture name. */
  name: string;
  /** Packed byte count, divisible by four. */
  byteLength: number;
  /** Typed-array interpretation of the captured words. */
  elementType: GPUComputationCaptureElementType;
  /** Number of words per logical value. */
  components?: number;
};

/** Coordinate and description of a snapshot frame. */
export type GPUSnapshotLabel = {
  /** Iteration coordinate; defaults to the frame index. */
  iteration?: number;
  /** Last graph node before this frame's copies. */
  afterNode?: string;
  /** Human-readable description, stored as capture `meaning`. */
  label?: string;
};

/** Copies frames into one bounded readback buffer, then maps that buffer once. */
export class GPUSnapshotArena {
  /** Device owning the readback buffer. */
  readonly device: Device;
  /** Prefix for capture identifiers. */
  readonly id: string;
  /** Ordered ranges in every frame. */
  readonly slots: readonly GPUSnapshotSlot[];
  /** Bytes of one snapshot frame (all slots, each four-byte aligned). */
  readonly frameByteLength: number;
  /** Maximum number of frames. */
  readonly capacity: number;

  private readonly buffer: Buffer;
  private readonly labels: GPUSnapshotLabel[] = [];
  private readback?: Promise<Uint8Array>;
  private destroyed = false;

  /** Allocates at most `maximumByteLength` bytes (64 MiB by default). */
  constructor(
    device: Device,
    props: {
      id?: string;
      slots: readonly GPUSnapshotSlot[];
      capacity: number;
      maximumByteLength?: number;
    }
  ) {
    this.device = device;
    this.id = props.id ?? 'gpu-snapshot-arena';
    this.capacity = props.capacity;
    this.slots = props.slots.map(slot => Object.freeze({...slot}));
    const names = new Set<string>();
    for (const slot of this.slots) {
      this.validateByteRange(slot.byteLength, `slot "${slot.name}" byteLength`);
      if (names.has(slot.name)) throw new Error(`${this.id} duplicate slot "${slot.name}"`);
      names.add(slot.name);
      if (
        slot.components !== undefined &&
        (!Number.isSafeInteger(slot.components) ||
          slot.components < 1 ||
          slot.byteLength % (slot.components * 4) !== 0)
      ) {
        throw new Error(`${this.id} slot "${slot.name}" must contain whole components`);
      }
    }
    this.frameByteLength = this.slots.reduce((total, slot) => total + slot.byteLength, 0);
    const byteLength = this.capacity * this.frameByteLength;
    if (!Number.isSafeInteger(this.capacity) || this.capacity < 0) {
      throw new Error(`${this.id} capacity must be a non-negative integer`);
    }
    const maximumByteLength = props.maximumByteLength ?? 64 * 1024 * 1024;
    if (
      !Number.isSafeInteger(byteLength) ||
      !Number.isSafeInteger(maximumByteLength) ||
      maximumByteLength < 0 ||
      byteLength > maximumByteLength
    ) {
      throw new Error(`${this.id} exceeds maximumByteLength; decimate snapshot frames`);
    }
    this.buffer = device.createBuffer({
      id: this.id,
      byteLength,
      usage: Buffer.MAP_READ | Buffer.COPY_DST
    });
  }

  /** Number of frames recorded so far. */
  get count(): number {
    return this.labels.length;
  }

  /** Records every slot into the next frame and returns its zero-based index. */
  encodeSnapshot(
    commandEncoder: CommandEncoder,
    sources: Readonly<Record<string, {buffer: Buffer; byteOffset?: number}>>,
    label: GPUSnapshotLabel = {}
  ): number {
    if (this.destroyed || this.readback) throw new Error(`${this.id} is closed for snapshots`);
    if (this.count >= this.capacity) throw new Error(`${this.id} snapshot arena is full`);
    // Validate the entire frame before recording any copies.
    for (const slot of this.slots) {
      const source = sources[slot.name];
      if (!source) throw new Error(`${this.id} missing source for slot "${slot.name}"`);
      this.validateByteRange(source.byteOffset ?? 0, `slot "${slot.name}" byteOffset`);
      if ((source.byteOffset ?? 0) + slot.byteLength > source.buffer.byteLength) {
        throw new Error(`${this.id} source for slot "${slot.name}" is too small`);
      }
    }
    const frame = this.count;
    let destinationOffset = frame * this.frameByteLength;
    for (const slot of this.slots) {
      const source = sources[slot.name];
      if (slot.byteLength > 0) {
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: source.buffer,
          sourceOffset: source.byteOffset ?? 0,
          destinationBuffer: this.buffer,
          destinationOffset,
          size: slot.byteLength
        });
      }
      destinationOffset += slot.byteLength;
    }
    this.labels.push({...label});
    return frame;
  }

  /**
   * Maps once after all frames have been submitted. Each capture owns a copy of its values.
   * `runId` belongs to the enclosing capture sidecar, not individual capture entries.
   */
  async read(
    props: {
      runId?: string;
      graph?: string;
      resources?: Readonly<Record<string, string>>;
      keySpaces?: Readonly<Record<string, string>>;
    } = {}
  ): Promise<GPUComputationCapture[]> {
    if (this.destroyed) throw new Error(`${this.id} has been destroyed`);
    this.readback ??=
      this.count && this.frameByteLength
        ? this.buffer.readAsync()
        : Promise.resolve(new Uint8Array(0));
    const bytes = await this.readback;
    const captures: GPUComputationCapture[] = [];
    for (const [frame, label] of this.labels.entries()) {
      let byteOffset = frame * this.frameByteLength;
      for (const slot of this.slots) {
        const copy = bytes.slice(byteOffset, byteOffset + slot.byteLength);
        const Values =
          slot.elementType === 'f32'
            ? Float32Array
            : slot.elementType === 'i32'
              ? Int32Array
              : Uint32Array;
        const resource = props.resources?.[slot.name];
        const keySpace = props.keySpaces?.[slot.name];
        captures.push({
          id: `${this.id}-${slot.name}-${frame}`,
          name: slot.name,
          iteration: label.iteration ?? frame,
          elementType: slot.elementType,
          values: new Values(copy.buffer, copy.byteOffset, copy.byteLength / 4),
          provenance: 'gpu-readback',
          ...(props.graph !== undefined && {graph: props.graph}),
          ...(resource !== undefined && {resource}),
          ...(keySpace !== undefined && {keySpace}),
          ...(label.afterNode !== undefined && {afterNode: label.afterNode}),
          ...(label.label !== undefined && {meaning: label.label}),
          ...(slot.components !== undefined && {components: slot.components})
        });
        byteOffset += slot.byteLength;
      }
    }
    return captures;
  }

  /** Releases the readback buffer. Already returned values remain valid. */
  destroy(): void {
    if (!this.destroyed) this.buffer.destroy();
    this.destroyed = true;
  }

  private validateByteRange(value: number, name: string): void {
    if (!Number.isSafeInteger(value) || value < 0 || value % 4 !== 0) {
      throw new Error(`${this.id} ${name} must be a non-negative multiple of 4`);
    }
  }
}
