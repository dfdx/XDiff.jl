
using Compat
using Iterators
using Espresso
import Espresso: to_expr, expr, canonical, canonical_calls, to_context, OpName
import Espresso: reduce_equalities, flatten, @get

include("utils.jl")
include("ops.jl")
include("deriv.jl")
include("rules.jl")
include("tderiv.jl")
include("trules.jl")
include("rdiff.jl")
include("tdiff.jl")

