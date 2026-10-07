// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {PickingInfo, Viewport} from '@deck.gl/core';
import type {CommandEncoder} from '@luma.gl/core';

/** Camera a scene starts from. Longitude/latitude in degrees, standard Web Mercator zoom. */
export type ViewState = {
  longitude: number;
  latitude: number;
  zoom: number;
  pitch?: number;
  bearing?: number;
};

/** Per-frame values passed to a scene instance's `encode`. */
export type SceneFrame = {
  /** Deck viewport used for this frame. */
  viewport: Viewport;
  /** Monotonic wall-clock time in seconds (`performance.now() / 1000`). */
  timeSeconds: number;
  /** Seconds since the previous encoded frame, clamped to `[0, 0.25]`. */
  deltaSeconds: number;
  /** Number of frames encoded since this scene instance was created. */
  frameIndex: number;
};

/** Pointer event forwarded to a scene. `coordinate` is `[longitude, latitude]` when known. */
export type ScenePointerEvent = {
  info: PickingInfo;
  /** `[longitude, latitude]` under the pointer, or `null` off the map. */
  coordinate: readonly [number, number] | null;
  /** Pointer position in CSS pixels relative to the deck canvas. */
  pixel: readonly [number, number];
  /** Modifier keys held at the time of the event. */
  shiftKey: boolean;
  altKey: boolean;
  ctrlKey: boolean;
  metaKey: boolean;
};

/** The part of a scene instance the deck effect needs. */
export type EncodableInstance = {
  encode: (commandEncoder: CommandEncoder, frame: SceneFrame) => void;
};
