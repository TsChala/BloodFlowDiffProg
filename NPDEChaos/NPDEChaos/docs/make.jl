using Documenter, SeqData

makedocs(sitename="SeqData", modules=[SeqData], doctest=true,
pages = [
    "Home" => "index.md"]
    )

deploydocs(
    repo = "github.com/maximilian-gelbrecht/SeqData.jl.git",
    )
