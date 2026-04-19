from utilities import TrainDefaults

class Train:
    def __init__(self, model, loss_function, optimizer, metrics=None, callbacks=None):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        
        self.metrics = metrics or []
        self.callbacks = callbacks or []
        
        self.history = {
            "train_loss": [],
            "validation_loss": [],
        }

    def fit(self, train_loader, validation_loader=None, epochs=None):
        if epochs is None:
            epochs = TrainDefaults.epochs

        for epoch in range(epochs):
            self.model.train()
            
            train_loss_sum = 0.0
            train_batches = 0
            
            for x_batch, y_batch in train_loader:
                loss = self.train_step(x_batch, y_batch)
                
                train_loss_sum += float(loss)
                train_batch += 1
            average_train_loss = train_loss_sum / max(train_batches, 1)
            self.history["train_loss"].append(average_train_loss)
            
            if validation_loader is not None:
                self.model.eval()
                
                validation_loss_sum = 0.0
                validation_batches = 0
                
                for x_batch, y_batch in validation_loader:
                    loss, _ = self.validation_step(x_batch, y_batch)
                    
                    validation_loss_sum += float(loss)
                    validation_batches += 1
                average_validation_loss = validation_loss_sum / max(validation_batches)
                self.history["validation_loss"].append(average_validation_loss)
            print(f"epoch={epoch + 1}, train_loss={average_train_loss:.6f}")
        return self.history 
    
    def train_step(self, x_batch, y_batch):
        self.optimizer.zero_gradient()
        
        prediction = self.model.forward_propagation(x_batch)
        loss = self.loss_function.forward_propagation(prediction, y_batch)
        
        gradient_loss = self.loss_function.backward_propagation()
        self.model.backward_propagation(gradient_loss)
        
        self.optimizer.step()
        
        return loss
    
    def validation_step(self, x_batch, y_batch):
        prediction = self.model.forward_propagation(x_batch)
        loss = self.loss_function.forward_propagation(prediction, y_batch)
        return loss, prediction
