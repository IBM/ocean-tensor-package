import ocean

idx = ocean.index[[1,2,3],:,...,-2]
idx2 = ocean.gpu[0](idx)
ocean.cpu(idx2, True)

idx3 = idx2.setDevice(ocean.gpu[0])
idx2.setDevice(ocean.gpu[0],True)

