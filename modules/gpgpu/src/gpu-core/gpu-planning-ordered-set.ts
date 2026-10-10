// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

type GPUPlanningOrderedSetNode<Value> = {
  value: Value;
  priority: number;
  insertionOrder: number;
  left?: GPUPlanningOrderedSetNode<Value>;
  right?: GPUPlanningOrderedSetNode<Value>;
};

/** Deterministic treap supporting allocator best-fit queries in expected logarithmic time. @internal */
export class GPUPlanningOrderedSet<Value> {
  private root?: GPUPlanningOrderedSetNode<Value>;
  private insertionOrder = 0;

  constructor(private readonly compare: (left: Value, right: Value) => number) {}

  add(value: Value): void {
    const insertionOrder = this.insertionOrder++;
    const treeNode: GPUPlanningOrderedSetNode<Value> = {
      value,
      priority: getPlanningPriority(insertionOrder),
      insertionOrder
    };
    this.root = this.insert(this.root, treeNode);
  }

  /** Removes and returns the smallest value greater than or equal to `minimum`. */
  takeLowerBound(minimum: Value): Value | undefined {
    let treeNode = this.root;
    let lowerBound: GPUPlanningOrderedSetNode<Value> | undefined;
    while (treeNode) {
      if (this.compare(treeNode.value, minimum) >= 0) {
        lowerBound = treeNode;
        treeNode = treeNode.left;
      } else {
        treeNode = treeNode.right;
      }
    }
    if (!lowerBound) return undefined;
    this.root = this.remove(this.root, lowerBound.value);
    return lowerBound.value;
  }

  /** Removes and returns the largest value. */
  takeMaximum(): Value | undefined {
    let treeNode = this.root;
    if (!treeNode) return undefined;
    while (treeNode.right) treeNode = treeNode.right;
    this.root = this.remove(this.root, treeNode.value);
    return treeNode.value;
  }

  private insert(
    root: GPUPlanningOrderedSetNode<Value> | undefined,
    treeNode: GPUPlanningOrderedSetNode<Value>
  ): GPUPlanningOrderedSetNode<Value> {
    if (!root) return treeNode;
    const order = this.compare(treeNode.value, root.value);
    if (order === 0) {
      root.value = treeNode.value;
      return root;
    }
    if (order < 0) {
      root.left = this.insert(root.left, treeNode);
      if (hasHigherPriority(root.left, root)) root = rotatePlanningTreeRight(root);
    } else {
      root.right = this.insert(root.right, treeNode);
      if (hasHigherPriority(root.right, root)) root = rotatePlanningTreeLeft(root);
    }
    return root;
  }

  private remove(
    root: GPUPlanningOrderedSetNode<Value> | undefined,
    value: Value
  ): GPUPlanningOrderedSetNode<Value> | undefined {
    if (!root) return undefined;
    const order = this.compare(value, root.value);
    if (order < 0) {
      root.left = this.remove(root.left, value);
      return root;
    }
    if (order > 0) {
      root.right = this.remove(root.right, value);
      return root;
    }
    return mergePlanningTrees(root.left, root.right);
  }
}

function hasHigherPriority<Value>(
  candidate: GPUPlanningOrderedSetNode<Value> | undefined,
  current: GPUPlanningOrderedSetNode<Value>
): boolean {
  return Boolean(
    candidate &&
      (candidate.priority < current.priority ||
        (candidate.priority === current.priority &&
          candidate.insertionOrder < current.insertionOrder))
  );
}

function rotatePlanningTreeLeft<Value>(
  root: GPUPlanningOrderedSetNode<Value>
): GPUPlanningOrderedSetNode<Value> {
  const replacement = root.right!;
  root.right = replacement.left;
  replacement.left = root;
  return replacement;
}

function rotatePlanningTreeRight<Value>(
  root: GPUPlanningOrderedSetNode<Value>
): GPUPlanningOrderedSetNode<Value> {
  const replacement = root.left!;
  root.left = replacement.right;
  replacement.right = root;
  return replacement;
}

function mergePlanningTrees<Value>(
  left: GPUPlanningOrderedSetNode<Value> | undefined,
  right: GPUPlanningOrderedSetNode<Value> | undefined
): GPUPlanningOrderedSetNode<Value> | undefined {
  if (!left) return right;
  if (!right) return left;
  if (hasHigherPriority(left, right)) {
    left.right = mergePlanningTrees(left.right, right);
    return left;
  }
  right.left = mergePlanningTrees(left, right.left);
  return right;
}

/** Integer mixing keeps insertion-ordered inputs from degenerating the treap. */
function getPlanningPriority(insertionOrder: number): number {
  let value = (insertionOrder + 1) >>> 0;
  value = Math.imul(value ^ (value >>> 16), 0x7feb352d);
  value = Math.imul(value ^ (value >>> 15), 0x846ca68b);
  return (value ^ (value >>> 16)) >>> 0;
}
