max_marginal_relevance_search(query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, filter: Callable | Dict[str, Any] | None = None, **kwargs: Any) → List[Document][source]
Return docs selected using the maximal marginal relevance.

Maximal marginal relevance optimizes for similarity to query AND diversity among selected documents.

Parameters
:
query (str) – Text to look up documents similar to.

k (int) – Number of Documents to return. Defaults to 4.

fetch_k (int) – Number of Documents to fetch before filtering (if needed) to pass to MMR algorithm.

lambda_mult (float) – Number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to minimum diversity. Defaults to 0.5.

filter (Callable | Dict[str, Any] | None)

kwargs (Any)

Returns
:
List of Documents selected by maximal marginal relevance.