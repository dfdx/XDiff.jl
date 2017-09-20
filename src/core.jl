
# core.jl - single place to load all package definitions.
#
# If you are willing to learn the package structure, just go through
# included files one by one, read header notes and other comments

using IterTools
using Einsum
using Espresso
importall Espresso
import Espresso: canonical, canonical_calls, to_context, OpName, without_indices
import Espresso: reduce_equalities, flatten, @get, @get_or_create, isspecial



include("utils.jl")
include("ops.jl")
include("deriv.jl")
include("rules.jl")
include("tderiv.jl")
include("trules.jl")
include("special.jl")
include("specialrules.jl")
include("rdiff.jl")
include("tdiff.jl")
