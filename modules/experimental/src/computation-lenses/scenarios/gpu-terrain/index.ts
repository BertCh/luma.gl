// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Registers the gpu-terrain lens scenarios (side-effect imports): the pack, then one scenario
// list per family. Each family file is appended to by one implementer batch.
import '../../../gpu-terrain/computation-pack/index';
import './analysis.scenarios';
import './point-horizon.scenarios';
import './decode.scenarios';
import './features.scenarios';
import './geomorphometry.scenarios';
import './illumination.scenarios';
import './relief.scenarios';
import './hydrology.scenarios';
import './flow-field.scenarios';
