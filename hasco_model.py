class RobertaForSequenceClassification_hasoc(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.output_hidden_states = True
        # add cnn params

        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.gru= OnLSTM(768, 512, num_layers=1, bidirectional=True, batch_first=True).cuda()

        # self.gru = nn.ModuleList(self.gru)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768*3, config.num_labels)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooler_out = outputs[1]
        sequence_output_1 = outputs[2][-1]


        x1 = torch.mean(sequence_output_1, 1)

        x_max, _= torch.max(sequence_output_1, 1)

        cat_out = torch.cat((x1,  x_max, pooler_out), 1)


        # by fc
        logits = self.classifier(cat_out)


        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = [loss, ]
            outputs = outputs + [nn.functional.softmax(logits, -1)]
        else:
            outputs = nn.functional.softmax(logits, -1)

        return outputs  # (loss), logits, (hidden_states), (attentions)

