
using Compat
using Iterators
using Espresso
import Espresso: expr, canonical, canonical_calls, to_context, flatten, @get
import Espresso: reduce_equalities

include("utils.jl")
include("ops.jl")
include("deriv.jl")
include("rules.jl")
include("tderiv.jl")
include("trules.jl")
include("rdiff.jl")
include("tdiff.jl")
