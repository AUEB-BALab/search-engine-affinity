from functools import wraps
import numpy as np
import click
from seanalysis import utils
from seanalysis.controller import controller as ctrl
from seanalysis.compare_sorting import find_distance
from seanalysis.drawing.visualization import Visualization


def handle_exception(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except utils.SEAnalysisException as e:
            raise click.UsageError(click.style(str(e), fg='red'))
    return wrapper


def validate_query_categories(categories, all_categories):
    if categories is None:
        return all_categories
    categories = categories.split(',')
    invalid_categories = set(categories).difference(all_categories)
    if invalid_categories:
        raise click.UsageError('Unknown query category %s' %
                               invalid_categories.pop())
    return categories


def validate_index_option(index, per_day, model):
    if index and per_day:
        raise click.UsageError('--index is not supported by --per-day option.')
    if index and model != 'se':
        raise click.UsageError('--index option is only supported with "se"'
                               ' model')
    return index, per_day, model


def validate_weights(weights):
    for i, w in enumerate(weights):
        weights[i] = (
            np.asarray(w.split(','), dtype=np.float)
            if isinstance(w, str)
            else w
        )
    if not len(weights[0]) == len(weights[1]) == len(weights[2]):
        raise click.UsageError('The weights a, b, c should have the same'
                               ' length.')
    return weights[0], weights[1], weights[2]


@click.group()
@click.option('--config', help='Location of configuration file',
              type=click.Path(), envvar='SESIM_CONFIG')
@click.option('--search_engines', '-s', help='Search engines for the analysis',
              type=str, required=True)
@click.option('--categories', help='Categories of the queries',
              type=str)
@click.option('-N', help='Number of results for each query', default=10,
              type=int)
@click.option('--merge', help='Display analysis of query categories in a'
              ' single diagram', default=False, is_flag=True)
@click.pass_context
@handle_exception
def sesim(ctx, config, search_engines, categories, n, merge):
    conf = utils.load_config(config)
    query_categories = conf['categories']
    context = {
        'config_file': conf,
        'results': conf['results'],
        'search_engines': search_engines.split(','),
        'categories': validate_query_categories(
            categories, query_categories.keys()),
        'N': n,
        'merge': merge
    }
    ctx.obj = context


def _extract_context(ctx):
    query_categories = ctx.obj.get('categories')
    search_engines = ctx.obj.get('search_engines')
    results_dir = ctx.obj.get('results')
    merge = ctx.obj.get('merge')
    n = ctx.obj.get('N')
    config = ctx.obj.get('config_file')
    return (query_categories, search_engines, results_dir, merge, n, config)


@sesim.command()
@click.option('--method', help='Method to use for analyzing search engines'
              ' similarity', type=click.Choice(['cmp', 'clf']), required=True)
@click.option('--model', '-m', help='The model in which data will be'
              ' transformed',
              type=click.Choice(['tensor', 'lda', 'query', 'se']),
              required=True)
@click.option('config', '-c', help='The model specific configuration',
              type=str, required=True, multiple=True)
@click.option('--vocabulary', help='Number of words of vocabulary to be used'
              ' for the' 'analysis. If not specified all terms compose'
              ' vocabulary (top terms)', default=None, type=int)
@click.option('--per-day/--per-result', default=False)
@click.option('--index', help='Run an a classifier for the index'
              ' classification problem. It is only available with'
              ' --per-result option and "se" model.', default=False,
              is_flag=True)
@click.pass_context
@handle_exception
def cont(ctx, method, model, config, vocabulary, per_day, index):
    (query_categories, search_engines, results_dir, merge,
            n, conf_file) = _extract_context(ctx)
    index, per_day, model = validate_index_option(index, per_day, model)
    snippets = {}
    queries = {}
    for category in query_categories:
        queries[category] = conf_file['categories'][
            category]
        snippets[category] = utils.load_snippets(
            results_dir, category, search_engines, N=n, per_day=per_day)
    controller = ctrl.Controller(method.lower(), model.lower(), config)
    controller.analyze(snippets, index, vocabulary, merge)


@sesim.command()
@click.option('--metric', help='Metric used for computing similarity',
              type=click.Choice(
                  ['LEV', 'DAM-LEV', 'HAM', 'JAR', 'JAR-WIN', 'PYTHON',
                   'KENDALL', 'SPEARMAN', 'G', 'M']), required=True)
@click.pass_context
@handle_exception
def rank(ctx, metric):
    (query_categories, search_engines, results_dir, merge,
            n, _) = _extract_context(ctx)
    visual = Visualization(merge, len(query_categories))
    for query_category in query_categories:
        urls = utils.load_urls(
            results_dir, query_category, search_engines, N=n)
        find_distance(visual, urls, None, None, search_engines,
                      query_category, metric, None, n, None,
                      None, None)
    visual.show()


@sesim.command()
@click.option('--evol', help='Display the evolutiion similarity of two search'
              ' engines over time', default=False, is_flag=True)
@click.option('weight_a', '-a', help='The weight of transpositions.', type=str,
              default=[0.8])
@click.option('weight_b', '-b', help='The weight of snippets.', type=str,
              default=[1])
@click.option('weight_c', '-c', help='The weight of titles.', type=str,
              default=[0.33])
@click.pass_context
@handle_exception
def metrict(ctx, evol, weight_a, weight_b, weight_c):
    (query_categories, search_engines, results_dir, merge,
            n, _) = _extract_context(ctx)
    categories = '*' if evol else query_categories
    weight_a, weight_b, weight_c = validate_weights([weight_a, weight_b,
                                                     weight_c])
    visual = Visualization(merge, len(query_categories))
    for query_category in categories:
        urls = utils.load_urls(
            results_dir, query_category, search_engines, N=n)
        snippets = utils.load_snippets(
                results_dir, query_category, search_engines, N=n,
                per_day=False)
        titles = utils.load_titles(
            results_dir, query_category, search_engines, N=n,
            per_day=False)
        find_distance(visual, urls, snippets, titles, search_engines,
                      query_category, 'T', evol, n, weight_a,
                      weight_b, weight_c)
    visual.show()


def main():
    sesim()
