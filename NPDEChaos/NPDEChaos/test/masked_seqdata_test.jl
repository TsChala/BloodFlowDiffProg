using SeqData, Distributions




A = rand(50, 10, 1000)




train, valid, test = SequentialData(A, 24, 64, 0.5, 0.25, verbose=true, supervised=true)

train = MaskedSequentialData(train, [1:10,:,:,:], 0., [2:2,5:10,:,:], nothing)

train[1][1]
train[1][2]
train[end][1]
train[end][2]


train, valid, test = SequentialData(A, 24, 64, 0.5, 0.25, verbose=true, supervised=false)

train = MaskedSequentialData(train, [1:10,:,:,:], 0., [2:2,5:10,:,:], nothing)

train[1]
train[end]


true
