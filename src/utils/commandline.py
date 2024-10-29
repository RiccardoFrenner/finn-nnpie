import argparse
from typing import Any, TypeVar, get_origin, get_args, Union, Optional
import dataclasses as dc

T = TypeVar("T")


def dict_to_dataclass(cls: T, data: dict[str, Any], consume: bool) -> T:
    """Convert a dictionary to a dataclass instance."""
    fields = {f.name for f in cls.__dataclass_fields__.values()}
    filtered_data = {k: v for k, v in data.items() if k in fields}
    if consume:
        for k in filtered_data:
            del data[k]
    return cls(**filtered_data)


def create_argparse_argument(
    parser: argparse.ArgumentParser, name: str, type_info: Any, **kwargs
):
    assert isinstance(name, str), f"{type(name)=}, {name=}"

    origin = get_origin(type_info)
    args = get_args(type_info)

    if origin is list:
        inner_type = args[0]
        # TODO: "*" (allows 0 elements) may also be an option but not sure how to
        # pass this info into here and what kind of edge cases may result from that
        parser.add_argument(name, type=inner_type, nargs="+", **kwargs)
    elif origin is tuple:
        inner_types = args
        parser.add_argument(
            name,
            type=lambda s: tuple(
                [t(x) for t, x in zip(inner_types, s.split(","), strict=True)]
            ),
            **kwargs,
        )
    elif origin is dict:
        # Handle dict type
        key_type, value_type = args
        parser.add_argument(
            name,
            type=lambda s: {
                key_type(k): value_type(v)
                for k, v in (item.split("=") for item in s.split(","))
            },
            **kwargs,
        )
    elif origin is set:
        # Handle set type
        inner_type = args[0]
        parser.add_argument(
            name,
            type=lambda s: set(inner_type(item) for item in s.split(",")),
            **kwargs,
        )
    elif origin is Union:
        # If the type hint is a union, use type as a lambda function that checks the types
        def union_type(value):
            for arg in args:
                try:
                    return arg(value)
                except (ValueError, TypeError):
                    pass
            raise ValueError(f"Invalid value for union type: {value}")

        parser.add_argument(name, type=union_type, **kwargs)
    elif origin is Optional:
        inner_type = args[0]
        assert kwargs["default"] is None
        parser.add_argument(name, type=inner_type, **kwargs)
        print("="*100)
    else:
        # Handle basic types
        if type_info is bool and "action" in kwargs:
            parser.add_argument(name, **kwargs)
        else:
            parser.add_argument(name, type=origin or type_info, **kwargs)


def parser_from_dataclasses(dataclasses, postfixes: Optional[list[str]]=None, positional_args: Optional[set[str]] = None):
    postfixes = ["" for _ in dataclasses] if postfixes is None else postfixes
    positional_args = set() if positional_args is None else set(positional_args)

    parser = argparse.ArgumentParser()
    for postfix, dataclass in zip(postfixes, dataclasses):
        group = parser.add_argument_group(f"{dataclass.__name__}{postfix}")
        for f in dc.fields(dataclass):
            field_name = f.name
            field_type = f.type

            default_value = dc.MISSING
            if f.default is not dc.MISSING:
                default_value = f.default
            if f.default_factory is not dc.MISSING:
                default_value = f.default_factory()

            if default_value is not dc.MISSING:
                help_text = f" (default: {default_value})"
            else:
                help_text = ""

            argument_name = (
                f"{field_name}" if field_name in positional_args else f"--{field_name}"
            )
            argument_kwargs = dict(
                default=default_value,
                help=help_text,
            )
            if default_value is dc.MISSING:
                argument_kwargs.pop("default")

            # if it's None, we should handle it like other types
            if field_type is bool and default_value is not None:
                argument_kwargs["default"] = default_value
                if default_value:
                    argument_kwargs["action"] = "store_false"
                    argument_kwargs["help"] = f"Disable {field_name}"
                    argument_kwargs["dest"] = field_name
                    argument_name = f"--no-{field_name}"
                    assert (
                        field_name not in positional_args
                    ), f"{field_name} should not be in {positional_args}"
                else:
                    argument_kwargs["action"] = "store_true"

            # print(argument_name, field_type, argument_kwargs)
            create_argparse_argument(
                group, f"{argument_name}{postfix}", field_type, **argument_kwargs
            )

    return parser


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    # # Define types
    # a = list[int]
    # b = tuple[int, str]
    # c = dict[str, float]
    # d = set[str]
    # e = int

    # # Create arguments dynamically
    # create_argparse_argument(parser, "a", a)
    # create_argparse_argument(parser, "b", b)
    # create_argparse_argument(parser, "c", c)
    # create_argparse_argument(parser, "d", d)
    # create_argparse_argument(parser, "e", e)

    # parser.print_help()

    # # Parse arguments
    # args = parser.parse_args(
    #     "0 1 2 3 4 5,hello key1=1.0,key2=2.0 item1,item2,item3,item3 10".split(" ")
    # )
    # print(args)
    # print(type(args.b[0]))
    # print(type(args.b[1]))

    @dc.dataclass
    class TestClass:
        a: float
        b: list[int]
        c: int = 6
        d: str = "test334"
        e: list[int] = dc.field(default_factory=list)
        f: list[int] = dc.field(default_factory=lambda: [4, 5, 6, 7])
        g: tuple[float, int] = dc.field(default_factory=lambda: (1.2, 66))

    parser2 = parser_from_dataclasses([TestClass], positional_args={"d", "a"})
    parser2.print_help()
    args2 = parser2.parse_args("4 1,2,3,4".split(" "))
    print(args2)
    print(TestClass(**vars(args2)))

    # parser2 = argparse.ArgumentParser()
    # parser2.add_argument("--a", required=True, default=3)
    # parser2.add_argument("--b", required=True)
    # parser2.print_help()
