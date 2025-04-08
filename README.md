Einops 'rearrange' Implementation from Scratch
This notebook implements a subset of the 'einops' library's 'rearrange' functionality using only Python and NumPy.

Approach:
The implementation follows a multi-step process:

1. Parsing ('_parse_pattern', '_split_top_level', '_parse_composition'):

- The pattern string is split using a robust method ('_split_top_level') that respects parentheses, allowing for basic nesting like '(a (b c))'.
- Ellipsis ('...') can be placed anywhere in the pattern (but must appear consistently on both sides). Its position is recorded relative to the original specifications.
- Input and output specifications (like 'h', '(h w)', '((a b) c)') are processed to generate flattened lists of elementary identifiers (e.g., '((a b) c)' becomes ['a', 'b', 'c']) for permutation and internal logic. The original structured specifications are also retained to correctly map tensor dimensions during input analysis.
- Basic validation of identifiers and parenthesis balance occurs.

2. Input Analysis:

- Tensor dimensions are matched against the input specifications (considering the original structure and ellipsis position) to identify prefix, ellipsis, and suffix dimensions.
- A dictionary 'axis_sizes' maps each elementary input axis identifier to its calculated size.
- Splitting operations (e.g., '(h w)' on input) are handled by analyzing the structured components, calculating sizes using the corresponding tensor dimension size and any provided 'axes_lengths'.

3. Intermediate Reshape:

- The input tensor is reshaped (if necessary) into an "intermediate" form where dimensions correspond to the elementary axes identified in the input decomposition, preserving the relative order of ellipsis dimensions.
- (Note: Reshaping when ellipsis is in the middle of the pattern might be restricted if it requires complex dimension reordering that 'np.reshape' cannot handle directly).

4. Output Analysis & Permutation:

- The output pattern's structure is analyzed to determine the shape elements of the final tensor, including calculating merged dimensions ('(...)' on output) and identifying axes for repetition (new axes specified in 'axes_lengths').
- A permutation map is calculated using the flattened elementary axis lists for input and output. This map specifies how to reorder the axes of the 'intermediate_tensor' (including the ellipsis dimensions) to match the desired output decomposition order.

5.Transpose:

- 'np.transpose' applies the calculated permutation to the (potentially reshaped) intermediate tensor.

6.Repetition:

- For each genuinely new axis introduced in the output (present in 'axes_lengths' but not originating from an input axis), 'np.expand_dims' and 'np.repeat' are used to insert and tile this new dimension at the correct location in the permuted tensor. The insertion index is determined based on the final output decomposition order.
- Note on Semantics: This implementation uses tiling for new axes ('a b -> a b c'). It does not strictly enforce the 'einops' behavior of erroring if trying to "repeat" an existing axis that wasn't size 1 (e.g., 'a b c -> a d c' where b=2, d=5). It relies on NumPy's reshape/transpose behavior for such cases, which might error if sizes mismatch or succeed (effectively renaming) if they match.

7.Final Reshape (Merging):

- The tensor (now transposed and with repeated axes added) is reshaped one last time, if necessary, to achieve the final target shape. This step handles merging axes specified by '(...)'' in the output pattern and ensures the ellipsis dimensions are in their correct final place.
8. Error Handling: Includes checks for invalid patterns, shape mismatches, inconsistent/missing 'axes_lengths', parenthesis mismatch, invalid identifiers, unused input axes, and NumPy operation errors (reshape, transpose, repeat).

Design Decisions & Improvements
- Robust Parser: Uses a depth-counting splitter ('_split_top_level') and recursive composition parsing ('_parse_composition') to handle basic nesting like '(a (b c))'. Returns both original specs and flattened decomposition for different stages of the logic.
- Flexible Ellipsis: Ellipsis ('...') is supported anywhere in the pattern (start, middle, end), provided it's consistent on both sides. The core logic handles shape calculations, permutation, and final reshaping around the ellipsis.
- Intermediate Decomposition: Uses the strategy of decomposing specifications into elementary axes to determine sizes and permutations, simplifying the handling of combined operations like split-transpose-merge.
- Repetition Behavior: Implements repetition by tiling new axes ('expand_dims' + 'repeat'). This is slightly more permissive than strict 'einops' for cases involving replacing existing non-1 dimensions but handles the common use case of adding/tiling new axes.

Known Limitations
- Complex Nesting/Parsing: While basic nesting '(a (b c))' is handled, extremely complex or ambiguous patterns might not parse correctly. The parser flattens nesting during decomposition, which might lose structural information needed for some hypothetical advanced patterns.
- Anonymous '1' Mapping: Mapping input dimensions of size 1 specified via the literal '1' to output '1's relies on order and availability, which could be ambiguous in patterns with multiple '1's (e.g., '1 a 1 -> 1 1 a').
- Performance: Relies on standard 'NumPy' operations ('reshape', 'transpose', 'repeat'), which may involve intermediate data copies and might not be as optimized as the backend-specific implementations in the actual 'einops' library, especially for very large tensors.
- Strict Repeat Semantics: Does not strictly error when "replacing" a non-1 dimension with a sized dimension (e.g., 'a b c -> a d c' where b!=1), relying instead on 'NumPy's' behavior which might raise a 'ValueError' during reshape/transpose if sizes are incompatible.

How to Run Tests
1. Execute the Setup, Parser, and Core Implementation cells in order.
2. Execute the Unit Tests cell
