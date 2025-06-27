from swift.plugin.loss_scale.loss_scale import (
    LossScale,
    loss_scale_map,
    ContextType,
    Messages,
)


class PerTurnLossScale(LossScale):
    """Weight comes from each assistant message's `my_weight` field."""

    def __call__(
        self, context_list: list[str], context_types: list[ContextType], messages: Messages, **kwargs
    ) -> tuple[list[str], list[float]]:
        res_context_list = []
        res_loss_scale = []
        i = 0
        n_round = len(messages) // 2
        for context, context_type in zip(context_list, context_types):
            is_last_round = i + 1 == n_round
            loss_weight = 1.0
            if context_type == ContextType.RESPONSE:
                assert context == messages[2 * i + 1]["content"]
                loss_weight = messages[2 * i + 1].get("loss_weight", 1.0)
                i += 1
            new_context, loss_scale = self.get_loss_scale(
                context,
                context_type,
                is_last_round,
                weight=float(loss_weight),
            )
            res_context_list += new_context
            res_loss_scale += loss_scale
        return res_context_list, res_loss_scale

    def get_loss_scale(self, context, context_type, is_last_round, *, weight=1.0, **_):
        return [context], [float(weight)]


loss_scale_map["per_turn_weight"] = PerTurnLossScale
