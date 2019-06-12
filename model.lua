--- Load up network model or initialize from scratch
local opts = require 'opts'
image = require 'image'
local opt = opts.parse(arg)
paths.dofile('models/discriminator.lua')
paths.dofile('opts.lua')

-- Continuing an experiment where it left off
optimStateD = {
      learningRate = opt.LR_D,
      learningRateDecay = opt.LRdecay_D,
      momentum = opt.momentum_D,
      weightDecay = opt.weightDecay_D,
      alpha = opt.alpha_D,
      epsilon = opt.epsilon_D
}

if not finalPredictions then
    if opt.netD ~= 'none' then
        assert(paths.filep(opt.netD), 'File not found: ' .. opt.netD)
        print('==> Loading model from: ' .. opt.netD)
        netD = torch.load(opt.netD)
    else
        print ('createModelD')
        netD = createModelD()
    end
end

-- Criterion (can be set in the opt.task file as well)

if not finalPredictions then
    criterionD_real = nn[opt.crit .. 'Criterion']()

    criterionG2 = nn.ParallelCriterion()
    criterionG2:add(nn[opt.crit .. 'Criterion'](), opt.lambda_G)
end

if opt.GPU ~= -1 then
    -- Convert model to CUDA
    print('==> Converting model to CUDA')
    --model:cuda()
    --criterion:cuda()

    if not finalPredictions then
        print ('netD')
        netD:cuda()
        criterionG2:cuda()
        criterionD_real:cuda()
    end

    cudnn.fastest = true
    cudnn.benchmark = true
end

k_t = opt.init_Kt
measure = 0.0
