using Documenter, PaddedMatricesForwardDiff

makedocs(;
    modules=[PaddedMatricesForwardDiff],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/chriselrod/PaddedMatricesForwardDiff.jl/blob/{commit}{path}#L{line}",
    sitename="PaddedMatricesForwardDiff.jl",
    authors="Chris Elrod",
    assets=String[],
)

deploydocs(;
    repo="github.com/chriselrod/PaddedMatricesForwardDiff.jl",
)
