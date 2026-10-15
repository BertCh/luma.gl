// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Registers the gpu-raster analysis lens scenarios (side-effect imports): the pack, then one
// scenario list per family. Each family file is appended to by one implementer batch.
import '../../../gpu-raster/computation-pack/index';
import './cost-distance.scenarios';
import './zones.scenarios';
import './algebra.scenarios';
import './isolines.scenarios';
import './sampling.scenarios';
import './vector-field.scenarios';
import './regions.scenarios';
