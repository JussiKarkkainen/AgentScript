agent WorldModel {
    environment: CarRacing-v2
    train: True
    dataset: None
    
    
    Modules {
        Perception {
            model: VAE()
            vision_resolution: [1, 3, 64, 64]
            laten_dim: 32
            enc_kernel_size: 4
            dec_kernel_size: 5
            text: None
            audio: None
        }

        WorldModel {
            planning: One_Step
        }
    }
}

