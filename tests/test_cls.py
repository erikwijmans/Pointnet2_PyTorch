import pytest


@pytest.mark.parametrize("use_xyz", ["True", "False"])
@pytest.mark.parametrize("model", ["ssg", "msg"])
def test_cls(use_xyz, model):
    model = pytest.helpers.get_model(
        ["task=cls", f"model={model}", f"model.use_xyz={use_xyz}"]
    )
    pytest.helpers.cls_test(model)
