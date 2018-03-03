import keras
import torch

class Model(object):
    def __init__(self, model_instance):
        self.model_instance = model_instance

        if isinstance(model_instance, keras.models.Model):
            self.library = "keras"
        elif isinstance(model_instance, torch.nn.Module):
            self.library = "pytorch"
        else:
            raise NotImplementedError("Model type: '{}' is not supported!".format(
                type(model_instance))
            )

    def compile(self, num_epochs, steps_per_epoch, optimizer, **optimizer_args):
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        if self.library == "pytorch":
            # TODO: Make more general by including other optimizers at a later point

            self.optimizer = torch.optim.SGD(
                self.model_instance.parameters(), lr=0.1,
                momentum=0.9, nesterov=True, weight_decay=5e-4
            )

            self.scheduler = MultiStepLR(
                self.optimizer, milestones=[60, 120, 160], gamma=0.2
            )
        elif self.library == "keras":
            return self.model_instance.compile(
                loss=optimizer_args["loss"],
                optimizer=optimizer,
                metrics=["accuracy"]
            )
        else:
            raise NotImplementedError()


    def fit_generator(self, augmentation_generator,  x_validation, y_validation):
        if self.library == "pytorch":
            criterion = torch.nn.CrossEntropyLoss().cuda()

            for epoch in range(self.epochs):
                progress_bar = tqdm(
                    islice(augmentation_generator, self.steps_per_epoch)
                )

                for i, (images, labels) in enumerate(progress_bar):
                    progress_bar.set_description("Epoch " + str(epoch))

                    images = Variable(images).cuda(async=True)
                    labels = Variable(labels).cuda(async=True)

                    self.model_instance.grad()
                    pred = self.model_instance(images)

                    xentropy_loss = criterion(pred, labels)
                    xentropy_loss.backward()
                    self.optimizer.step()

                    xentropy_loss_avg += xentropy_loss.data[0]

                    # Calculate running average of accuracy
                    _, pred = torch.max(pred.data, 1)
                    total += labels.size(0)
                    correct += (pred == labels.data).sum()
                    accuracy = correct / total

                    progress_bar.set_postfix(
                        xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
                        acc='%.3f' % accuracy
                    )
                # XXX: Compute test accuracy and step scheduler (see train.py)
                # XXX: Make this return a keras style history object
                raise NotImplementedError()

                self.scheduler.step(epoch)


        elif self.library == "keras":
            return self.model_instance.fit_generator(
                augmentation_generator, steps_per_epoch=self.steps_per_epoch,
                epochs = self.num_epochs + 1,
                validation_data=(x_validation, y_validation),
                initial_epoch=self.num_epochs
            )
        else:
            raise NotImplementedError()

    def evaluate(self, x_validation, y_validation, *args, **kwargs):
        if self.library == "pytorch":

            self.model_instance.eval()  # change model to "eval" mode (BN uses moving mean/var)
            correct, total = 0., 0.

            for images, labels in zip(x_validation, y_validation):
                images = Variable(images, volatile=True).cuda()
                labels = Variable(labels, volatile=True).cuda()

                pred = cnn(images)

                pred = torch.max(pred.data, 1)[1]
                total += labels.size(0)
                correct += (pred == labels.data).sum()

            val_acc = correct / total
            self.model_instance.train()
            return val_acc
        elif self.library == "keras":
            return self.model_instance.evaluate(x_validation, y_validation, verbose=0)
        else:
            raise NotImplementedError()
