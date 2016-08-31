

def read_meter_data(trace_filename, project_info_filename,
                    project_id=None, weather=True, merge_series=True):
    """Read meter data from a raw XML file source, obtain matching project
    information from a separate CSV file.  Fetches the corresponding weather
    data, when requested, too.

    Parameters
    ==========
    trace_filename: str
        Filename of XML meter trace.
    project_info_filename: str
        Filename of CSV file containing project info.
    project_id: str
        Manually provide the project ID used in `project_info_filename`.
        If `None`, the first part of `trace_filename` before a `_` is used.
    weather: bool
        `True` will obtain weather (temperature) data.
    merge_series: bool
        `True` will return a `pandas.DataFrame` with merged consumption and
        temperature data.

    Returns
    =======
    A `DataCollection` object with the following fields:
        project_info: `pandas.DataFrame`
            Contains columns for project properties.
        baseline_end: `pandas.Datetime`
            End date of the baseline period.
        consumption_data: `eemeter.consumption.ConsumptionData`
            Consumption data object.
        consumption_data_freq: `pandas.DataFrame`
            Consumption data with normalized frequency.
    If :samp:`weather=True`:
        weather_source: `eemeter.ISDWeatherSource`
            Weather source object.
        weather_data: `pandas.DataFrame`
            Temperature observations in, degF, with frequency matching
            `consumption_data_freq`.  Values are averaged if raw temperature
            observations are lower frequency.
    If :samp:`merge_series=True`:
        cons_weather_data: `pandas.DataFrame`
            Merged consumption and temperature data.
    """
    from eemeter.meter import DataCollection, DataContainer
    from eemeter.parsers import ESPIUsageParser

    # TODO: New API.
    #from eemeter.structures import (
    #    EnergyTrace,
    #    EnergyTraceSet,
    #    Intervention,
    #    ZIPCodeSite,
    #    Project
    #)
    #from eemeter.io.parsers import ESPIUsageParser
    from eemeter.weather import ISDWeatherSource
    import pandas as pd
    import os

    with open(trace_filename, 'r') as f:
        parser = ESPIUsageParser(f.read())

    consumption_datas = list(parser.get_consumption_data_objects())
    cons_data_obj = consumption_datas[0]

    all_projects_info = pd.read_csv(project_info_filename)
    fuel_type_map = {'electricity': 'E', 'natural_gas': 'NG'}

    if project_id is None:
        project_id = os.path.basename(trace_filename).split("_")[0]

    fuel_type = fuel_type_map[cons_data_obj.fuel_type]
    project_info = all_projects_info.query('project_id == "{}" and\
                                           fuel_type == "{}"'.format(
                                               project_id, fuel_type))

    baseline_end = pd.to_datetime(project_info.baseline_period_end.tolist()[0],
                                  utc=True)

    # Sometimes the data have differing observation frequencies,
    # so choose the most common one (in the usage data) and align
    # everything to that.
    cons_index_diff = cons_data_obj.data.index.to_series().diff(periods=1)
    new_freq = pd.value_counts(cons_index_diff).argmax()
    cons_data = cons_data_obj.data.tz_convert("UTC")
    cons_data = cons_data.resample(new_freq).mean()

    res = DataCollection(project_info=project_info,
                         baseline_end=baseline_end,
                         consumption_data=cons_data_obj,
                         consumption_data_freq=cons_data)
    if weather:
        station = unicode(project_info.weather_station.tolist()[0])
        ws = ISDWeatherSource(station)

        res.add_data(DataContainer("weather_source", ws, None))

        ws.add_year_range(cons_data.index.min().year,
                          cons_data.index.max().year)

        weather_data = ws._unit_convert(ws.tempC, "degF").tz_localize("UTC")
        weather_data = weather_data.resample(new_freq).mean()

        res.add_data(DataContainer("weather_data", weather_data, None))

        if weather_data.empty:
            raise ValueError("No weather data")

        if merge_series:
            cons_weather_data = pd.concat([cons_data, weather_data], axis=1,
                                          join="inner")
            cons_weather_data.columns = ['usage', 'temp']

            res.add_data(DataContainer("cons_weather_data", cons_weather_data,
                                       None))

    return res
