/** FCoSE layout parameter types and utilities — shared between GraphControls and consumers. */

export interface FcoseParams {
  nodeRepulsion: number
  idealEdgeLength: number
  gravity: number
}

export const DEFAULT_FCOSE_PARAMS: FcoseParams = {
  nodeRepulsion: 4500,
  idealEdgeLength: 50,
  gravity: 0.25,
}

/** Convert FcoseParams to Cytoscape layout options. */
export function toLayoutOptions(params: FcoseParams): Record<string, unknown> {
  return {
    nodeRepulsion: () => params.nodeRepulsion,
    idealEdgeLength: () => params.idealEdgeLength,
    gravity: params.gravity,
  }
}
